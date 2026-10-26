from __future__ import annotations

import json
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from typing_extensions import NotRequired, TypedDict

from conftest import ContentDecider
from ctxsieve import WindowPolicy
from ctxsieve.integrations.langgraph import WINDOW_KEY, SlidingWindowHook
from ctxsieve.window import is_note


class S(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ctxsieve_window: NotRequired[dict]


TURNS = [
    "q0 the task: plan a trip",
    "q1 PIN budget is 100",
    "q2 DROP nice weather today",
    "q3 what about hotels",
    "q4 DROP lol",
    "q5 PIN no flights after 9pm",
    "q6 summarize so far",
    "q7 final answer please",
]


def build(hook, checkpointer, store=None, seen=None):
    def sieve(state: S, runtime: Runtime):
        return hook.update(state, runtime.store)

    def model(state: S):
        view = hook.view(state)
        if seen is not None:
            seen.append(view)
        n = sum(isinstance(m, HumanMessage) and not is_note(m) for m in view)
        return {"messages": [AIMessage(f"reply to {state['messages'][-1].text[:2]} (saw {n} user msgs)")]}

    g = StateGraph(S)
    g.add_node("sieve", sieve)
    g.add_node("model", model)
    g.add_edge(START, "sieve")
    g.add_edge("sieve", "model")
    g.add_edge("model", END)
    return g.compile(checkpointer=checkpointer, store=store)


def run_turns(graph, turns, config):
    for t in turns:
        graph.invoke({"messages": [HumanMessage(t)]}, config)
    return graph.get_state(config).values


def test_view_mode_keeps_history_and_shows_pruned_view():
    decider, seen = ContentDecider(), []
    hook = SlidingWindowHook(decider, WindowPolicy(max_messages=4), mode="view")
    graph = build(hook, InMemorySaver(), seen=seen)
    config = {"configurable": {"thread_id": "t1"}}
    state = run_turns(graph, TURNS, config)

    # History untouched: every turn plus every reply, no notes.
    msgs = state["messages"]
    assert [m.text for m in msgs if isinstance(m, HumanMessage)] == TURNS
    assert len(msgs) == 2 * len(TURNS) and not any(is_note(m) for m in msgs)

    view = seen[-1]
    texts = [m.text for m in view]
    assert "q0 the task: plan a trip" == texts[0]  # first human is head
    assert "q1 PIN budget is 100" in texts and "q5 PIN no flights after 9pm" in texts
    assert not any("DROP" in t for t in texts)
    assert sum("dropped" in t for t in texts if t.startswith("[ctxsieve")) == 2
    assert len(view) < len(msgs)
    # The last model call ran before the final reply existed; its window is the 4 newest messages then.
    assert [m.id for m in view[-4:]] == [m.id for m in msgs[-5:-1]]

    # Each evicted unit was asked about exactly once across all turns.
    asked = sum(len(c) for c in decider.calls)
    window_state = state[WINDOW_KEY]
    json.dumps(window_state)  # checkpoint-friendly
    evicted = sum(len(e["items"]) for e in window_state["region"])
    assert asked == evicted


def test_view_mode_resumes_from_checkpoint_without_reasking():
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t2"}}
    first = ContentDecider()
    run_turns(build(SlidingWindowHook(first, WindowPolicy(max_messages=4)), saver), TURNS[:6], config)

    # A fresh process: new hook, new decider, same checkpointer.
    second, seen = ContentDecider(), []
    state = run_turns(build(SlidingWindowHook(second, WindowPolicy(max_messages=4)), saver, seen=seen), TURNS[6:], config)
    asked_again = sum(len(c) for c in second.calls)
    # Two more turns add 4 messages, so at most 4 new units leave the window.
    assert 0 < asked_again <= 4
    assert [m.text for m in state["messages"] if isinstance(m, HumanMessage)] == TURNS
    assert "q1 PIN budget is 100" in [m.text for m in seen[-1]]


def test_state_mode_rewrites_messages_and_archives_to_store():
    store, seen = InMemoryStore(), []
    hook = SlidingWindowHook(ContentDecider(), WindowPolicy(max_messages=4), mode="state")
    graph = build(hook, InMemorySaver(), store=store, seen=seen)
    config = {"configurable": {"thread_id": "t3"}}
    state = run_turns(graph, TURNS, config)

    msgs = state["messages"]
    texts = [m.text for m in msgs]
    assert not any("DROP" in t for t in texts)  # really gone from state
    assert any(is_note(m) and "dropped" in m.text for m in msgs)
    assert "q1 PIN budget is 100" in texts
    # Each dropped single message became one pointer note, so the count is unchanged here.
    assert sum(is_note(m) for m in msgs) == 2
    # In state mode the model sees the state as is.
    assert [m.id for m in seen[-1]] == [m.id for m in msgs[:-1]]

    full = hook.restore(state, store)
    assert [m.text for m in full if isinstance(m, HumanMessage)] == TURNS
    assert len(full) == 2 * len(TURNS)
    assert any("DROP" in m.text for m in full)


def test_state_mode_without_archive_or_store_fails_loudly():
    hook = SlidingWindowHook(ContentDecider(), WindowPolicy(max_messages=1), mode="state")
    with pytest.raises(ValueError, match="state mode needs an archive"):
        hook.update({"messages": [HumanMessage("a", id="a"), HumanMessage("b", id="b")]})


def test_view_mode_rebuilds_when_history_was_rewritten():
    hook = SlidingWindowHook(ContentDecider(), WindowPolicy(max_messages=2))
    msgs = [SystemMessage("s", id="s"), HumanMessage("t", id="t")] + [HumanMessage(f"m{i}", id=f"m{i}") for i in range(5)]
    update = hook.update({"messages": msgs})
    # Someone else removed m0 from the history: the stored window no longer resolves.
    rewritten = [m for m in msgs if m.id != "m0"]
    update2 = hook.update({"messages": rewritten, WINDOW_KEY: update[WINDOW_KEY]})
    view = hook.view({"messages": rewritten, WINDOW_KEY: update2[WINDOW_KEY]})
    assert "m0" not in [m.id for m in view]
    assert [m.id for m in view][-2:] == ["m3", "m4"]


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        SlidingWindowHook(ContentDecider(), mode="other")
