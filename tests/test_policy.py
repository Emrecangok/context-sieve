from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from conftest import conversations
from ctxsieve import InMemoryArchive, PrunePolicy, prune, restore
from ctxsieve.deciders import FakeDecider
from ctxsieve.policy import MARK, candidate_indices, choose_action, protected_indices

POLICY = PrunePolicy(preserve_recent=2, truncate_head_chars=50)
probs = st.floats(0.0, 1.0)


def scripted(p_verbatim: float, p_gist: float) -> FakeDecider:
    return FakeDecider(lambda q: p_verbatim if q.id.endswith(":verbatim") else p_gist)


@given(conversations(), probs, probs)
def test_only_tool_results_change_and_pairs_survive(msgs, pv, pg):
    out = prune(msgs, scripted(pv, pg), InMemoryArchive(), POLICY).messages
    assert len(out) == len(msgs)
    for before, after in zip(msgs, out):
        assert type(before) is type(after)
        assert before.id == after.id
        if not isinstance(before, ToolMessage):
            assert before == after
        else:
            assert before.tool_call_id == after.tool_call_id


@given(conversations(), probs, probs)
def test_protected_messages_untouched(msgs, pv, pg):
    policy = PrunePolicy(preserve_recent=2, truncate_head_chars=50, pinned_ids=frozenset({"m3", "m5"}))
    out = prune(msgs, scripted(pv, pg), InMemoryArchive(), policy).messages
    for i in protected_indices(msgs, policy):
        assert out[i] == msgs[i]
    for i, m in enumerate(msgs):
        if isinstance(m, SystemMessage) or m.id in {"m3", "m5"} or i >= len(msgs) - 2:
            assert out[i] == msgs[i]
    first_human = next(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    assert out[first_human] == msgs[first_human]


@given(conversations())
def test_fail_open_on_decider_error(msgs):
    res = prune(msgs, FakeDecider(error=RuntimeError("boom")), InMemoryArchive(), POLICY)
    assert res.messages == msgs
    if candidate_indices(msgs, POLICY):
        assert res.error is not None
        assert all(r.action == "keep" and r.reason == "decider_error" for r in res.receipts)


@given(conversations())
def test_missing_answers_keep(msgs):
    res = prune(msgs, FakeDecider(None), InMemoryArchive(), POLICY)
    assert res.messages == msgs


@given(conversations(), probs, probs)
def test_restore_round_trip(msgs, pv, pg):
    archive = InMemoryArchive()
    res = prune(msgs, scripted(pv, pg), archive, POLICY)
    assert restore(res.messages, archive) == msgs


@given(conversations(), probs, probs)
def test_idempotent(msgs, pv, pg):
    decider = scripted(pv, pg)
    once = prune(msgs, decider, InMemoryArchive(), POLICY).messages
    twice = prune(once, decider, InMemoryArchive(), POLICY).messages
    assert twice == once


@given(conversations(), probs, probs)
def test_one_receipt_per_candidate(msgs, pv, pg):
    res = prune(msgs, scripted(pv, pg), InMemoryArchive(), POLICY)
    assert [r.index for r in res.receipts] == candidate_indices(msgs, POLICY)
    for r in res.receipts:
        assert (r.archive_key is None) == (r.action == "keep")


@settings(max_examples=50)
@given(conversations(), st.floats(POLICY.act_below, 1.0))
def test_uncertain_means_keep(msgs, p):
    # Anything at or above the action threshold on "verbatim" keeps the message as is.
    res = prune(msgs, scripted(p, 0.0), InMemoryArchive(), POLICY)
    assert res.messages == msgs


def test_choose_action_table():
    pol = PrunePolicy(act_below=0.2)
    assert choose_action(0.9, 0.0, pol) == ("keep", "may_need_verbatim")
    assert choose_action(0.2, 0.0, pol) == ("keep", "may_need_verbatim")
    assert choose_action(0.1, 0.5, pol) == ("truncate", "gist_only")
    assert choose_action(0.1, 0.1, pol) == ("clear", "not_needed")
    assert choose_action(None, 0.0, pol) == ("keep", "no_answer")
    assert choose_action(0.1, None, pol) == ("keep", "no_answer")


def _example():
    long = "x" * 500
    return [
        SystemMessage("sys", id="s"),
        HumanMessage("find the weather", id="h"),
        AIMessage("", tool_calls=[{"id": "a", "name": "search", "args": {}}], id="ai1"),
        ToolMessage(long, tool_call_id="a", name="search", id="t1"),
        AIMessage("", tool_calls=[{"id": "b", "name": "search", "args": {}}], id="ai2"),
        ToolMessage(long, tool_call_id="b", name="search", id="t2"),
        AIMessage("done", id="ai3"),
        HumanMessage("thanks", id="h2"),
    ]


def test_actions_on_concrete_conversation():
    msgs = _example()
    decider = FakeDecider(lambda q: {"a:verbatim": 0.05, "a:gist": 0.9, "b:verbatim": 0.05, "b:gist": 0.05}[q.id])
    archive = InMemoryArchive()
    res = prune(msgs, decider, archive, PrunePolicy(preserve_recent=2, truncate_head_chars=100))

    t1, t2 = res.messages[3], res.messages[5]
    assert t1.text.startswith("x" * 100) and "truncated from 500 chars" in t1.text
    assert t1.additional_kwargs[MARK]["action"] == "truncate"
    assert t2.text.startswith("[ctxsieve: cleared 500 chars")
    assert [r.action for r in res.receipts] == ["truncate", "clear"]
    assert len(archive) == 2
    assert restore(res.messages, archive) == msgs


def test_malformed_probabilities_keep():
    for bad in (None, "0.1", float("nan"), -0.5, 2.0, True):
        msgs = _example()
        res = prune(msgs, FakeDecider(lambda q, bad=bad: bad), InMemoryArchive(), PrunePolicy(preserve_recent=2))
        assert res.messages == msgs, bad
        assert all(r.action == "keep" for r in res.receipts), bad


def test_no_candidates_means_no_decider_call():
    decider = FakeDecider()
    msgs = [HumanMessage("hi", id="h"), AIMessage("hello", id="a")]
    assert prune(msgs, decider, InMemoryArchive()).messages == msgs
    assert decider.calls == []
