"""Regressions for the LangGraph integration review findings."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import NotRequired, TypedDict

from conftest import ContentDecider, FakeSummarizer
from ctxsieve import PrunePolicy, SummaryPolicy, WindowPolicy
from ctxsieve.deciders import FakeDecider
from ctxsieve.integrations.langchain import SlidingWindowMiddleware, ToolPruningMiddleware
from ctxsieve.integrations.langgraph import PRUNE_KEY, WINDOW_KEY, SlidingWindowHook, ToolPruneHook
from ctxsieve.window import is_note
from test_integration_agent import TURNS, ToolLoopModel, assert_pairs_valid, search

DROP = {"drop": 0.95, "pin": 0.03, "summarize": 0.02}


class S(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ctxsieve_window: NotRequired[dict]


def window_graph(hook, store=None, seen=None):
    def sieve(state: S, runtime: Runtime):
        return hook.update(state, runtime.store)

    def model(state: S):
        view = hook.view(state)
        if seen is not None:
            seen.append(view)
        return {"messages": [AIMessage(f"reply to {state['messages'][-1].text[:3]}")]}

    g = StateGraph(S)
    g.add_node("sieve", sieve)
    g.add_node("model", model)
    g.add_edge(START, "sieve")
    g.add_edge("sieve", "model")
    g.add_edge("model", END)
    return g.compile(checkpointer=InMemorySaver(), store=store)


def humans(msgs):
    return [m.text for m in msgs if isinstance(m, HumanMessage) and not is_note(m)]


def test_state_mode_rebuild_after_broken_state_keeps_notes_and_restore():
    store = InMemoryStore()
    hook = SlidingWindowHook(ContentDecider(), WindowPolicy(max_messages=4), mode="state",
                             summarizer=FakeSummarizer(), summary_policy=SummaryPolicy(flush_at=1))
    graph = window_graph(hook, store)
    config = {"configurable": {"thread_id": "r1"}}
    turns = ["q0 task", "q1 DROP secret code is 4711", "q2 more", "q3 more", "q4 more", "q5 more"]
    for t in turns:
        graph.invoke({"messages": [HumanMessage(t)]}, config)
    before = graph.get_state(config).values
    notes_before = [m.id for m in before["messages"] if is_note(m)]
    assert notes_before

    # Break the stored window: version mismatch (same path as an upgrade or a hand edit).
    graph.update_state(config, {WINDOW_KEY: {**before[WINDOW_KEY], "version": 999}})
    graph.invoke({"messages": [HumanMessage("q6 what was the code?")]}, config)
    after = graph.get_state(config).values
    assert set(notes_before) <= {m.id for m in after["messages"]}
    assert humans(hook.restore(after, store)) == turns + ["q6 what was the code?"]

    # A user edit removing a message still in the window.
    victim = next(m for m in after["messages"] if m.text == "q6 what was the code?")
    graph.update_state(config, {"messages": [RemoveMessage(id=victim.id)]})
    graph.invoke({"messages": [HumanMessage("q7 next")]}, config)
    final = graph.get_state(config).values
    assert set(notes_before) <= {m.id for m in final["messages"]}
    restored = humans(hook.restore(final, store))
    assert "q1 DROP secret code is 4711" in restored and "q7 next" in restored


def test_greeting_and_late_system_message_keep_their_order():
    store = InMemoryStore()
    hook = SlidingWindowHook(ContentDecider(), WindowPolicy(max_messages=20), mode="state")
    graph = window_graph(hook, store)
    config = {"configurable": {"thread_id": "r2"}}
    graph.invoke({"messages": [AIMessage("Hi! I'm your travel bot"), HumanMessage("q0 plan a trip")]}, config)
    graph.invoke({"messages": [SystemMessage("Reminder: EU timezone"), HumanMessage("q1 next")]}, config)
    texts = [m.text for m in graph.get_state(config).values["messages"]]
    assert texts == ["Hi! I'm your travel bot", "q0 plan a trip", "reply to q0 ",
                     "Reminder: EU timezone", "q1 next", "reply to q1 "]


def test_late_system_and_first_user_message_are_auto_pinned():
    hook = SlidingWindowHook(FakeDecider(DROP), WindowPolicy(max_messages=2))
    msgs = [AIMessage("greet", id="g"), HumanMessage("first", id="f"), SystemMessage("sys", id="s")]
    msgs += [HumanMessage(f"u{i}", id=f"u{i}") for i in range(4)]
    update = hook.update({"messages": msgs})
    view = hook.view({"messages": msgs, **update})
    ids = [m.id for m in view]
    assert "f" in ids and "s" in ids and "g" not in ids
    assert ids.index("f") < ids.index("s")


def test_prune_archive_failure_keeps_result_and_does_not_crash():
    class DownStore(InMemoryStore):
        def batch(self, ops):
            if any(type(op).__name__ == "PutOp" for op in ops):
                raise ConnectionError("store down")
            return super().batch(ops)

    model = ToolLoopModel()
    agent = create_agent(
        model, [search],
        middleware=[ToolPruningMiddleware(ContentDecider(), PrunePolicy(preserve_recent=2), mode="state")],
        checkpointer=InMemorySaver(), store=DownStore(),
    )
    config = {"configurable": {"thread_id": "r3"}}
    for t in TURNS:
        agent.invoke({"messages": [HumanMessage(t)]}, config)
    state = agent.get_state(config).values
    tools = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert all(m.text.startswith(("NOISE", "FACT")) for m in tools)  # nothing cleared, nothing lost
    assert state[PRUNE_KEY]["decided"] == {}  # archive errors are not cached


class ReusedIdModel(ToolLoopModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop, run_manager, **kwargs)
        msg = result.generations[0].message
        if msg.tool_calls:
            msg.tool_calls[0]["id"] = "call_0"
        return result


def test_reused_tool_call_ids_are_each_asked():
    decider = ContentDecider()
    agent = create_agent(
        ReusedIdModel(), [search],
        middleware=[ToolPruningMiddleware(decider, PrunePolicy(preserve_recent=2, truncate_head_chars=50))],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "r4"}}
    for t in ["find noise 1", "find fact 1", "find fact 2", "find fact 3"]:
        agent.invoke({"messages": [HumanMessage(t)]}, config)
    decided = agent.get_state(config).values[PRUNE_KEY]["decided"]
    assert sorted(decided.values()) == ["clear", "keep", "keep"]
    assert "call_0" not in decided  # keyed by message id


def test_messages_without_ids_are_left_alone():
    class P(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        ctxsieve_window: NotRequired[dict]
        ctxsieve_prune: NotRequired[dict]

    wh = SlidingWindowHook(FakeDecider(DROP), WindowPolicy(max_messages=2))
    th = ToolPruneHook(ContentDecider())

    def sieve(state: P):
        return {**wh.update(state), **th.update(state)}

    def model(state: P):
        assert wh.view(state) == state["messages"] and th.view(state) == state["messages"]
        return {"messages": [AIMessage("ok")]}

    g = StateGraph(P)
    g.add_node("sieve", sieve)
    g.add_node("model", model)
    g.add_edge(START, "sieve")
    g.add_edge("sieve", "model")
    g.add_edge("model", END)
    graph = g.compile()
    out = graph.invoke({"messages": [HumanMessage(f"m{i}") for i in range(6)]})
    assert len(out["messages"]) == 7


def test_adjacent_drops_render_as_one_note_and_view_stays_bounded():
    hook = SlidingWindowHook(FakeDecider(DROP), WindowPolicy(max_messages=4))
    msgs, state = [HumanMessage("task", id="t")], {}
    for i in range(60):
        msgs += [HumanMessage(f"u{i}", id=f"u{i}"), AIMessage(f"a{i}", id=f"a{i}")]
        state = {"messages": msgs, **hook.update({"messages": msgs, **state})}
    view = hook.view(state)
    assert len(view) == 1 + 1 + 4  # head, one merged pointer, window
    assert "116 earlier message(s) dropped" in view[1].text and "+111 more" in view[1].text


def test_view_keeps_unknown_messages_in_place():
    hook = SlidingWindowHook(FakeDecider(DROP), WindowPolicy(max_messages=4))
    msgs = [HumanMessage("task", id="t"), AIMessage("", tool_calls=[{"id": "c0", "name": "s", "args": {}}], id="a"),
            ToolMessage("r", tool_call_id="c0", id="tr")]
    state = {"messages": msgs, **hook.update({"messages": msgs})}
    injected = HumanMessage("INJECTED CONTEXT", id="inj")
    view = hook.view(state, [injected, *msgs])
    assert [m.id for m in view] == ["inj", "t", "a", "tr"]
    late = HumanMessage("late", id="late")
    assert [m.id for m in hook.view(state, [*msgs, late])] == ["t", "a", "tr", "late"]


class AStore(AsyncBatchedBaseStore):
    """Async-first store like AsyncPostgresStore: sync calls on the event loop are rejected."""

    def __init__(self):
        super().__init__()
        self.inner = InMemoryStore()

    async def abatch(self, ops):
        return self.inner.batch(ops)


def test_async_agent_with_async_store_in_state_mode():
    async def go():
        store = AStore()  # async stores bind to the running loop
        agent = create_agent(
            ToolLoopModel(), [search],
            middleware=[
                ToolPruningMiddleware(ContentDecider(), PrunePolicy(preserve_recent=2), mode="state"),
                SlidingWindowMiddleware(ContentDecider(), WindowPolicy(max_messages=6), mode="state"),
            ],
            checkpointer=InMemorySaver(), store=store,
        )
        config = {"configurable": {"thread_id": "r5"}}
        for t in TURNS:
            await agent.ainvoke({"messages": [HumanMessage(t)]}, config)
        return (await agent.aget_state(config)).values

    state = asyncio.run(go())
    cleared = [m for m in state["messages"] if isinstance(m, ToolMessage) and m.text.startswith("[ctxsieve: cleared")]
    assert cleared  # tool pruning archived through the async store
    assert_pairs_valid(state["messages"])
