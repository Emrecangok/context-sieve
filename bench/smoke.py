"""Live smoke test: a short conversation per method with real OpenAI models.

    python -m bench.smoke                 # all methods
    python -m bench.smoke ctxsieve trim   # some methods

Needs OPENAI_API_KEY (in the environment or .env) and the index from `python -m bench.build_index`.
Prints token usage per method; it checks that everything runs, it does not measure quality.
"""

from __future__ import annotations

import sys
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from bench.docs_agent import (
    METHODS,
    build_agent,
    load_index,
    load_env,
    openai_chat_model,
    openai_decider,
    openai_embeddings,
)

TURNS = [
    "Hi! I'm building a support bot with LangGraph and I use a Postgres checkpointer in production.",
    "How do I trim old messages so the model doesn't see the whole history?",
    "What does ClearToolUsesEdit do and what is its default trigger?",
    "Thanks. Unrelated: how do I stream tokens from an agent?",
    "Going back to my setup: which checkpointer did I say I use, and what is the class name for it?",
]


def run(method: str) -> dict:
    model = openai_chat_model()
    decider = openai_decider(prompt_cache_key=f"smoke-{method}") if method == "ctxsieve" else None
    agent = build_agent(method, model=model, store=load_index(openai_embeddings()), decider=decider,
                        checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"smoke-{method}"}}
    usage = {"input": 0, "output": 0}
    start = time.perf_counter()
    for t in TURNS:
        out = agent.invoke({"messages": [HumanMessage(t)]}, config)
    for m in out["messages"]:
        if isinstance(m, AIMessage) and m.usage_metadata:
            usage["input"] += m.usage_metadata.get("input_tokens", 0)
            usage["output"] += m.usage_metadata.get("output_tokens", 0)
    result = {
        "method": method,
        "seconds": round(time.perf_counter() - start, 1),
        "agent_tokens": usage,
        "final_answer": out["messages"][-1].text[:300],
    }
    if decider is not None:
        result["decider_usage"] = decider.usage
    return result


def main(argv: list[str]) -> int:
    load_env()
    for method in argv or METHODS:
        print(run(method))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
