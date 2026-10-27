"""Offline checks for the benchmark agent: snapshot integrity and every method's wiring."""

from __future__ import annotations

import hashlib
import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

pytest.importorskip("numpy")
pytest.importorskip("langchain_text_splitters")

from bench.docs_agent import (  # noqa: E402
    METHODS,
    SNAPSHOT,
    Budget,
    build_agent,
    build_index,
    chunk,
    load_index,
    load_snapshot,
)
from conftest import ContentDecider  # noqa: E402


def test_snapshot_is_intact_and_licensed():
    rows = [json.loads(line) for line in SNAPSHOT.read_text(encoding="utf-8").splitlines()]
    pages = [ln for ln in (SNAPSHOT.parent / "pages.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert [r["path"] for r in rows] == pages
    for r in rows:
        assert hashlib.sha256(r["text"].encode("utf-8")).hexdigest() == r["sha256"]
        assert r["chars"] == len(r["text"]) > 500
        assert "Was this page helpful" not in r["text"] and "Copy page" not in r["text"]
    assert "MIT License" in (SNAPSHOT.parent / "LICENSE").read_text(encoding="utf-8")


def test_chunks_are_deterministic():
    a, b = chunk(load_snapshot()), chunk(load_snapshot())
    assert [(c.id, c.page_content) for c in a] == [(c.id, c.page_content) for c in b]
    assert all(len(c.page_content) <= 2000 for c in a)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("idx") / "index.json"
    emb = DeterministicFakeEmbedding(size=64)
    built = build_index(emb, path)
    loaded = load_index(emb, path)
    assert len(loaded.store) == len(built.store)
    return loaded


class DocsLoopModel(BaseChatModel):
    """Searches the docs once per user turn, then answers; summarizes when asked to."""

    seen: list = Field(default_factory=list)
    n: int = 0

    @property
    def _llm_type(self) -> str:
        return "docs-loop"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any("Context Extraction Assistant" in m.text for m in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage("SUMMARY of earlier turns"))])
        self.seen.append(list(messages))
        if isinstance(messages[-1], HumanMessage):
            self.n += 1
            call = {"id": f"call_{self.n}", "name": "search_docs", "args": {"query": messages[-1].text}}
            return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(f"answer {self.n}"))])


TURNS = [f"question {i} about checkpointers and middleware" for i in range(6)]
SMALL = Budget(tokens=3000, messages=6, keep_tool_results=1)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_runs_end_to_end(method, store):
    model = DocsLoopModel()
    # Without a summarizer, "summarize" keeps text verbatim; drop by default to exercise eviction.
    agent = build_agent(method, model=model, store=store, decider=ContentDecider(default_choice="drop"),
                        budget=SMALL, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": method}}
    for t in TURNS:
        agent.invoke({"messages": [HumanMessage(t)]}, config)
    msgs = agent.get_state(config).values["messages"]
    tool_results = [m for m in msgs if isinstance(m, ToolMessage)]
    last_input = model.seen[-1]

    if method == "full":
        assert len(last_input) == len(msgs) - 1 + 1  # all history (+ system prompt, - final answer)
    if method == "summarization":
        assert any("SUMMARY" in m.text for m in msgs)
    else:
        assert len(tool_results) == len(TURNS)
        assert tool_results[0].text.startswith("[")  # search output starts with the page path
    if method == "context_editing":
        assert any(m.text == "[cleared]" for m in last_input if isinstance(m, ToolMessage))
    if method in ("trim", "ctxsieve"):
        assert len(last_input) < len(msgs)


def test_unknown_method_rejected(store):
    with pytest.raises(ValueError):
        build_agent("nope", model=DocsLoopModel(), store=store)
