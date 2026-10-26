from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from conftest import ContentDecider
from ctxsieve import PrunePolicy, WindowPolicy
from ctxsieve.integrations.langchain import SlidingWindowMiddleware, ToolPruningMiddleware
from ctxsieve.integrations.langgraph import PRUNE_KEY


class ToolLoopModel(BaseChatModel):
    """Calls `search` once per user message, then answers. Records every input it sees."""

    seen: list = Field(default_factory=list)
    n: int = 0

    @property
    def _llm_type(self) -> str:
        return "tool-loop"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolLoopModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        last = messages[-1]
        if isinstance(last, HumanMessage):
            self.n += 1
            q = "noise" if "noise" in last.text else "fact"
            msg = AIMessage("", tool_calls=[{"id": f"call_{self.n}", "name": "search", "args": {"q": q}}])
        else:
            msg = AIMessage(f"done {self.n}")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def search(q: str) -> str:
    """Search the web."""
    return ("NOISE " if q == "noise" else "FACT ") + "x" * 600


TURNS = ["find noise 1", "find fact 1", "find noise 2", "find fact 2", "find noise 3"]


def run(agent, config, turns=TURNS):
    for t in turns:
        agent.invoke({"messages": [HumanMessage(t)]}, config)
    return agent.get_state(config).values


def assert_pairs_valid(msgs):
    calls = set()
    for m in msgs:
        if isinstance(m, AIMessage):
            calls.update(tc["id"] for tc in m.tool_calls)
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in calls
    results = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
    for m in msgs:
        if isinstance(m, AIMessage):
            assert all(tc["id"] in results for tc in m.tool_calls)


def test_tool_pruning_view_mode():
    model, decider = ToolLoopModel(), ContentDecider()
    agent = create_agent(
        model, [search],
        middleware=[ToolPruningMiddleware(decider, PrunePolicy(preserve_recent=2, truncate_head_chars=50), mode="view")],
        checkpointer=InMemorySaver(),
    )
    state = run(agent, {"configurable": {"thread_id": "p1"}})

    tools = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tools) == len(TURNS) and all(len(m.text) > 600 for m in tools)  # state untouched
    last_input = model.seen[-1]
    cleared = [m for m in last_input if isinstance(m, ToolMessage) and m.text.startswith("[ctxsieve: cleared")]
    kept = [m for m in last_input if isinstance(m, ToolMessage) and m.text.startswith("FACT")]
    assert len(cleared) == 2 and len(kept) == 2  # noise 1, 2 cleared; latest noise is still recent
    asked = [q.id for c in decider.calls for q in c]
    assert len(asked) == len(set(asked))  # every result asked about once
    assert set(state[PRUNE_KEY]["decided"].values()) == {"keep", "clear"}
    for seen in model.seen:
        assert_pairs_valid(seen)


def test_tool_pruning_state_mode_with_store():
    model, store = ToolLoopModel(), InMemoryStore()
    agent = create_agent(
        model, [search],
        middleware=[ToolPruningMiddleware(ContentDecider(), PrunePolicy(preserve_recent=2), mode="state")],
        checkpointer=InMemorySaver(), store=store,
    )
    state = run(agent, {"configurable": {"thread_id": "p2"}})
    tools = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    cleared = [m for m in tools if m.text.startswith("[ctxsieve: cleared")]
    assert len(cleared) == 2
    key = cleared[0].additional_kwargs["ctxsieve"]["archive_key"]
    original = store.get(("ctxsieve", "archive"), key)
    assert original is not None and original.value["message"]["data"]["content"].startswith("NOISE")
    assert_pairs_valid(state["messages"])


def test_sliding_window_view_mode_in_agent():
    model, decider = ToolLoopModel(), ContentDecider()
    agent = create_agent(
        model, [search],
        middleware=[SlidingWindowMiddleware(decider, WindowPolicy(max_messages=4), mode="view")],
        checkpointer=InMemorySaver(),
    )
    state = run(agent, {"configurable": {"thread_id": "w1"}})
    assert len(state["messages"]) == 4 * len(TURNS)  # human, tool call, tool result, answer
    for seen in model.seen:
        assert_pairs_valid(seen)
    assert len(model.seen[-1]) < len(state["messages"])
    asked = sum(len(c) for c in decider.calls)
    evicted = sum(len(e["items"]) > 0 for e in state["ctxsieve_window"]["region"])
    assert asked == evicted


def test_sliding_window_state_mode_in_agent():
    model, store = ToolLoopModel(), InMemoryStore()
    mw = SlidingWindowMiddleware(ContentDecider(), WindowPolicy(max_messages=4), mode="state")
    agent = create_agent(model, [search], middleware=[mw], checkpointer=InMemorySaver(), store=store)
    state = run(agent, {"configurable": {"thread_id": "w2"}})
    assert len(state["messages"]) <= len(model.seen[-1]) + 1
    assert_pairs_valid(state["messages"])
    full = mw.hook.restore(state, store)
    assert [m.text for m in full if isinstance(m, HumanMessage)] == TURNS


def test_async_invoke_view_mode():
    model = ToolLoopModel()
    agent = create_agent(
        model, [search],
        middleware=[SlidingWindowMiddleware(ContentDecider(), WindowPolicy(max_messages=4))],
        checkpointer=InMemorySaver(),
    )

    async def go():
        config = {"configurable": {"thread_id": "a1"}}
        for t in TURNS:
            await agent.ainvoke({"messages": [HumanMessage(t)]}, config)
        return (await agent.aget_state(config)).values

    state = asyncio.run(go())
    assert len(state["messages"]) == 4 * len(TURNS)
    assert len(model.seen[-1]) < len(state["messages"])
