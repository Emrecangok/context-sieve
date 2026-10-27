"""Benchmark agent: answers questions about the LangChain docs snapshot with a search tool.

Every context-management method runs on the same agent, model and index:

    full             no context management (upper bound on tokens)
    trim             keep the most recent tokens with trim_messages (model input only)
    summarization    LangChain SummarizationMiddleware
    context_editing  LangChain ContextEditingMiddleware + ClearToolUsesEdit
    ctxsieve         context-sieve ToolPruningMiddleware + SlidingWindowMiddleware (view mode)

Models come from the environment so they can change without code edits:
CTXSIEVE_AGENT_MODEL (default gpt-6-luna), CTXSIEVE_EMBED_MODEL (default
text-embedding-3-small), CTXSIEVE_DECIDER_MODEL (default gpt-6-luna).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelRequest,
    SummarizationMiddleware,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.utils import trim_messages
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from ctxsieve import PrunePolicy, WindowPolicy
from ctxsieve.deciders.base import Decider
from ctxsieve.integrations.langchain import SlidingWindowMiddleware, ToolPruningMiddleware

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "bench" / "corpus" / "langchain_docs" / "snapshot.jsonl"
INDEX = ROOT / "data" / "langchain_docs_index.json"

METHODS = ("full", "trim", "summarization", "context_editing", "ctxsieve")

SYSTEM_PROMPT = (
    "You are a LangChain and LangGraph documentation assistant. Use the search_docs tool to look "
    "up anything you are not sure about, and answer from what it returns. Keep answers short and "
    "name the docs page you used. Remember what the user tells you about their own setup."
)


def agent_model_name() -> str:
    return os.environ.get("CTXSIEVE_AGENT_MODEL", "gpt-6-luna")


def embed_model_name() -> str:
    return os.environ.get("CTXSIEVE_EMBED_MODEL", "text-embedding-3-small")


def decider_model_name() -> str:
    return os.environ.get("CTXSIEVE_DECIDER_MODEL", "gpt-6-luna")


# --- corpus and index -------------------------------------------------------------------


def load_snapshot(path: Path = SNAPSHOT) -> list[Document]:
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        docs.append(Document(r["text"], metadata={"path": r["path"], "url": r["url"], "sha256": r["sha256"]}))
    return docs


def chunk(docs: Sequence[Document], size: int = 2000, overlap: int = 200) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_documents(list(docs))
    for i, c in enumerate(chunks):
        c.id = f"chunk-{i:05d}"
    return chunks


def build_index(embeddings: Embeddings, path: Path = INDEX) -> InMemoryVectorStore:
    """Embed the snapshot once and save the vectors, so later runs never re-embed."""
    store = InMemoryVectorStore.from_documents(chunk(load_snapshot()), embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(path))
    return store


def load_index(embeddings: Embeddings, path: Path = INDEX) -> InMemoryVectorStore:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with `python -m bench.build_index`")
    return InMemoryVectorStore.load(str(path), embeddings)


def make_search_tool(store: InMemoryVectorStore, k: int = 4):
    @tool
    def search_docs(query: str) -> str:
        """Search the LangChain / LangGraph Python docs. Returns the most relevant passages."""
        hits = store.similarity_search(query, k=k)
        return "\n\n---\n\n".join(f"[{d.metadata['path']}]\n{d.page_content}" for d in hits)

    return search_docs


# --- context management methods ---------------------------------------------------------


class TrimMiddleware(AgentMiddleware):
    """Baseline: send only the most recent `max_tokens` of history to the model (state untouched)."""

    def __init__(self, max_tokens: int) -> None:
        super().__init__()
        self.max_tokens = max_tokens

    def _trimmed(self, request: ModelRequest) -> ModelRequest:
        kept = trim_messages(
            request.messages,
            max_tokens=self.max_tokens,
            token_counter="approximate",
            strategy="last",
            start_on="human",
            end_on=("human", "tool"),
        )
        return request.override(messages=kept)

    def wrap_model_call(self, request, handler):
        return handler(self._trimmed(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._trimmed(request))


@dataclass(frozen=True)
class Budget:
    """Rough context budget shared by all methods, so they are compared at similar sizes."""

    tokens: int = 6000
    messages: int = 12  # sliding-window size for ctxsieve
    keep_tool_results: int = 3


def middleware_for(
    method: str,
    *,
    budget: Budget = Budget(),
    decider: Decider | None = None,
    summary_model: BaseChatModel | str | None = None,
) -> list[AgentMiddleware]:
    if method == "full":
        return []
    if method == "trim":
        return [TrimMiddleware(budget.tokens)]
    if method == "summarization":
        if summary_model is None:
            raise ValueError("summarization needs summary_model")
        return [SummarizationMiddleware(summary_model, trigger=("tokens", budget.tokens), keep=("messages", budget.messages // 2))]
    if method == "context_editing":
        return [ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=budget.tokens, keep=budget.keep_tool_results)])]
    if method == "ctxsieve":
        if decider is None:
            raise ValueError("ctxsieve needs a decider")
        return [
            ToolPruningMiddleware(decider, PrunePolicy(preserve_recent=2 * budget.keep_tool_results)),
            SlidingWindowMiddleware(decider, WindowPolicy(max_messages=budget.messages)),
        ]
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def build_agent(
    method: str,
    *,
    model: BaseChatModel,
    store: InMemoryVectorStore,
    decider: Decider | None = None,
    budget: Budget = Budget(),
    checkpointer: Any = None,
    extra_middleware: Sequence[AgentMiddleware] = (),
):
    return create_agent(
        model,
        [make_search_tool(store)],
        system_prompt=SYSTEM_PROMPT,
        middleware=[*extra_middleware, *middleware_for(method, budget=budget, decider=decider, summary_model=model)],
        checkpointer=checkpointer,
    )


# --- real OpenAI components (need OPENAI_API_KEY) ---------------------------------------


def openai_chat_model() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    # gpt-6 models accept temperature only with reasoning_effort="none"; that also keeps runs
    # cheap and as repeatable as the API allows.
    return ChatOpenAI(model=agent_model_name(), reasoning_effort="none", temperature=0)


def openai_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=embed_model_name())


def openai_decider(prompt_cache_key: str | None = None) -> Decider:
    from ctxsieve.deciders.openai_decisions import OpenAIDecisionsDecider

    return OpenAIDecisionsDecider(model=decider_model_name(), prompt_cache_key=prompt_cache_key)


def load_env() -> None:
    """Load a local .env if python-dotenv is installed (keys never live in the repo)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")
