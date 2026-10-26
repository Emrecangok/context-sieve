from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from conftest import FakeSummarizer, chunks, conversations
from ctxsieve import (
    InMemoryArchive,
    SlidingWindow,
    SummaryPolicy,
    WindowPolicy,
    compact_summaries,
    flush_pending,
)
from ctxsieve.deciders import FakeDecider
from ctxsieve.region import Dropped, Pending, Pinned
from ctxsieve.window import choose_eviction

POLICY = WindowPolicy(max_messages=4)
AUTO_PIN = {"pinned_by_user", "system_message", "first_user_message"}


@st.composite
def choice_probs(draw):
    w = [draw(st.floats(0.0, 1.0)) for _ in range(3)]
    total = sum(w) or 1.0
    return dict(zip(("drop", "pin", "summarize"), (x / total for x in w)))


def feed(window: SlidingWindow, msgs, cuts) -> None:
    for chunk in chunks(msgs, cuts):
        window.add(chunk)


def assert_chronological(region):
    last = -1
    for e in region:
        seqs = [s for s, _ in e.items]
        assert min(seqs) > last
        last = max(seqs)


def window_messages(w: SlidingWindow):
    return [m for u in w.window for _, m in u]


def assert_pairs_valid(rendered):
    seen_calls = set()
    for m in rendered:
        if isinstance(m, AIMessage):
            seen_calls.update(tc["id"] for tc in m.tool_calls)
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in seen_calls
    present = {m.tool_call_id for m in rendered if isinstance(m, ToolMessage)}
    for m in rendered:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                assert tc["id"] in present


cuts = st.lists(st.integers(0, 60), max_size=5)


@given(conversations(), cuts, choice_probs())
def test_restore_all_is_original_order(msgs, cut, probs):
    w = SlidingWindow(FakeDecider(probs), InMemoryArchive(), POLICY)
    feed(w, msgs, cut)
    assert w.restore_all() == msgs


@given(conversations(), cuts, choice_probs())
def test_tool_pairs_stay_together(msgs, cut, probs):
    # conftest emits each AI tool call followed by all of its results.
    w = SlidingWindow(FakeDecider(probs), InMemoryArchive(), POLICY)
    feed(w, msgs, cut)
    assert_pairs_valid(w.render())


@given(conversations(), cuts, choice_probs())
def test_window_bounded_and_untouched(msgs, cut, probs):
    w = SlidingWindow(FakeDecider(probs), InMemoryArchive(), POLICY)
    feed(w, msgs, cut)
    inside = window_messages(w)
    assert len(inside) <= POLICY.max_messages or len(w.window) == 1
    # The window is the verbatim tail of the non-head messages.
    head_ids = {m.id for _, m in w.head}
    tail = [m for m in msgs if m.id not in head_ids]
    assert inside == tail[len(tail) - len(inside):]


@given(conversations(), cuts, choice_probs())
def test_head_always_rendered(msgs, cut, probs):
    w = SlidingWindow(FakeDecider(probs), InMemoryArchive(), POLICY)
    feed(w, msgs, cut)
    rendered = w.render()
    for m in msgs:
        if isinstance(m, SystemMessage):
            assert m in rendered
    first_human = next(m for m in msgs if isinstance(m, HumanMessage))
    assert rendered[: len(w.head)] == [m for _, m in w.head]
    assert first_human in rendered


@given(conversations(), cuts)
def test_fail_open_keeps_everything_verbatim(msgs, cut):
    w = SlidingWindow(FakeDecider(error=RuntimeError("down")), InMemoryArchive(), POLICY)
    feed(w, msgs, cut)
    assert not any(isinstance(e, Dropped) for e in w.region)
    rendered_ids = {m.id for m in w.render()}
    assert {m.id for m in msgs} <= rendered_ids
    if w.receipts:
        assert w.errors
        assert all(r.action == "pin" and r.reason in {"decider_error", *AUTO_PIN} for r in w.receipts)


@given(conversations(), cuts, choice_probs())
def test_uncertain_never_drops(msgs, cut, probs):
    policy = WindowPolicy(max_messages=4, drop_min_prob=0.8)
    w = SlidingWindow(FakeDecider(probs), InMemoryArchive(), policy)
    feed(w, msgs, cut)
    if probs["drop"] < 0.8:
        assert not any(isinstance(e, Dropped) for e in w.region)


@given(conversations(), cuts, choice_probs())
def test_one_receipt_per_evicted_unit_and_one_call_per_add(msgs, cut, probs):
    decider = FakeDecider(probs)
    w = SlidingWindow(decider, InMemoryArchive(), POLICY)
    evicting_adds = 0
    for chunk in chunks(msgs, cut):
        evicting_adds += any(r.reason not in AUTO_PIN for r in w.add(chunk))
    assert len(decider.calls) == evicting_adds  # one batched decider call per evicting add()
    assert len(w.receipts) == len(w.region)
    for r, e in zip(w.receipts, w.region):
        kind = {"drop": Dropped, "pin": Pinned, "summarize": Pending}[r.action]
        assert isinstance(e, kind)
        assert (r.action == "drop") == bool(r.archive_keys)
    assert sum(len(c) for c in decider.calls) == len(w.receipts)


@given(conversations(), st.lists(st.integers(0, 60), min_size=1, max_size=5), choice_probs())
def test_region_is_append_only(msgs, cut, probs):
    w = SlidingWindow(FakeDecider(probs), InMemoryArchive(), POLICY)
    before = []
    for chunk in chunks(msgs, cut):
        w.add(chunk)
        assert w.region[: len(before)] == before
        before = list(w.region)


answers = st.lists(st.one_of(st.none(), choice_probs()), min_size=1, max_size=6)
pins = st.frozensets(st.sampled_from([f"m{i}" for i in range(40)]), max_size=6)


@given(conversations(), cuts, answers, pins, st.integers(1, 4), st.integers(5, 60))
def test_mixed_decisions_stay_chronological_and_restorable(msgs, cut, ans, pinned, flush_at, max_chars):
    it = itertools.cycle(ans)
    w = SlidingWindow(FakeDecider(lambda q: next(it)), InMemoryArchive(), WindowPolicy(max_messages=3, pinned_ids=pinned))
    s = FakeSummarizer()
    for chunk in chunks(msgs, cut):
        w.add(chunk)
        flush_pending(w, s, SummaryPolicy(flush_at=flush_at))
        compact_summaries(w, s, SummaryPolicy(max_summary_chars=max_chars))
        assert_chronological(w.region)
    # Mid-stream a tool call may still await its result, so pairs are checked once all arrived.
    assert_pairs_valid(w.render())
    assert w.restore_all() == msgs
    for r in w.receipts:
        if r.probabilities is None or r.probabilities["drop"] < 0.8:
            assert r.action != "drop"
        if any(i in pinned for i in r.message_ids):
            assert r.action == "pin"


BAD_PROBS = [
    {"drop": None, "pin": 0.5, "summarize": 0.5},
    {"drop": "0.9", "pin": 0.05, "summarize": 0.05},
    {"drop": float("nan"), "pin": 0.5, "summarize": 0.5},
    {"drop": 1.5, "pin": 0.0, "summarize": 0.0},
    {"drop": True, "pin": 0.0, "summarize": 0.0},
    ["drop", "pin", "summarize"],
]


def test_malformed_probabilities_never_lose_messages():
    # Review finding: a None probability crashed choose_eviction after the units had left the window.
    for bad in BAD_PROBS:
        w = SlidingWindow(FakeDecider(lambda q, bad=bad: bad), InMemoryArchive(), WindowPolicy(max_messages=1))
        msgs = [HumanMessage("task", id="t"), HumanMessage("keep me", id="k"), AIMessage("a1", id="a1"), AIMessage("a2", id="a2")]
        w.add(msgs)
        assert w.restore_all() == msgs, bad
        assert [r.reason for r in w.receipts] == ["no_answer", "no_answer"], bad
        assert all(isinstance(e, Pinned) for e in w.region), bad


def test_unexpected_decision_error_keeps_units():
    class Exploding(FakeDecider):
        def decide(self, state, questions):
            return [None]  # not a Decision: breaks the caller, not the decider call

    w = SlidingWindow(Exploding(), InMemoryArchive(), WindowPolicy(max_messages=1))
    msgs = _chat(3)
    w.add(msgs)
    assert w.restore_all() == msgs
    assert w.errors and all(r.reason == "decider_error" for r in w.receipts)


def test_archive_failure_keeps_unit_verbatim():
    class BrokenArchive(InMemoryArchive):
        def put(self, message):
            raise OSError("disk full")

    w = SlidingWindow(FakeDecider({"drop": 1.0, "pin": 0.0, "summarize": 0.0}), BrokenArchive(), WindowPolicy(max_messages=2))
    msgs = _chat(4)
    w.add(msgs)
    assert [r.reason for r in w.receipts] == ["archive_error", "archive_error"]
    assert all(isinstance(e, Pinned) for e in w.region)
    assert w.render() == msgs and w.restore_all() == msgs
    assert w.errors


@given(conversations(), cuts, answers, st.integers(1, 4))
def test_state_round_trip(msgs, cut, ans, flush_at):
    import json

    from ctxsieve.archive import MessageIdArchive

    it = itertools.cycle(ans)
    decider = FakeDecider(lambda q: next(it))
    archive = MessageIdArchive(msgs)
    w = SlidingWindow(decider, archive, POLICY)
    s = FakeSummarizer()
    for chunk in chunks(msgs, cut):
        w.add(chunk)
        flush_pending(w, s, SummaryPolicy(flush_at=flush_at))

    data = json.loads(json.dumps(w.to_state()))
    w2 = SlidingWindow.from_state(data, msgs, decider, archive, POLICY)
    assert w2.region == w.region
    assert w2.window == w.window and w2.head == w.head
    assert w2.render() == w.render()
    assert w2.restore_all() == w.restore_all() == msgs
    assert w2.to_state() == w.to_state()


def test_from_state_reports_missing_messages():
    import pytest

    from ctxsieve.archive import MessageIdArchive
    from ctxsieve.window import MissingMessageError

    msgs = _chat(4)
    w = SlidingWindow(FakeDecider({"drop": 0.0, "pin": 1.0, "summarize": 0.0}), MessageIdArchive(msgs), WindowPolicy(max_messages=2))
    w.add(msgs)
    with pytest.raises(MissingMessageError):
        SlidingWindow.from_state(w.to_state(), msgs[:3], FakeDecider(), MessageIdArchive(msgs))
    with pytest.raises(ValueError):
        SlidingWindow.from_state({**w.to_state(), "version": 99}, msgs, FakeDecider(), MessageIdArchive(msgs))


def test_to_state_needs_ids():
    import pytest

    w = SlidingWindow(FakeDecider(), InMemoryArchive(), WindowPolicy(max_messages=5))
    w.add([HumanMessage("no id")])
    with pytest.raises(ValueError):
        w.to_state()


def test_choose_eviction_table():
    pol = WindowPolicy(drop_min_prob=0.8, fallback="pin")
    assert choose_eviction({"drop": 0.9, "pin": 0.05, "summarize": 0.05}, pol) == ("drop", "not_needed")
    assert choose_eviction({"drop": 0.7, "pin": 0.2, "summarize": 0.1}, pol) == ("pin", "needs_verbatim")
    assert choose_eviction({"drop": 0.7, "pin": 0.1, "summarize": 0.2}, pol) == ("summarize", "gist_enough")
    assert choose_eviction(None, pol) == ("pin", "no_answer")
    assert choose_eviction({"drop": 1.0}, pol) == ("pin", "no_answer")
    assert choose_eviction(None, WindowPolicy(fallback="summarize")) == ("summarize", "no_answer")


def _chat(n):
    msgs = [SystemMessage("sys", id="s"), HumanMessage("task", id="h0")]
    for i in range(n):
        msgs.append(AIMessage(f"a{i}", id=f"a{i}") if i % 2 else HumanMessage(f"u{i}", id=f"u{i}"))
    return msgs


def test_drop_leaves_pointer_and_is_recallable():
    archive = InMemoryArchive()
    w = SlidingWindow(FakeDecider({"drop": 0.95, "pin": 0.03, "summarize": 0.02}), archive, WindowPolicy(max_messages=2))
    w.add(_chat(4))
    rendered = w.render()
    assert [m.id for m in rendered[:2]] == ["s", "h0"]
    notes = [m for m in rendered if "dropped" in m.text]
    assert len(notes) == 1 and "2 earlier message(s) dropped" in notes[0].text  # adjacent drops merge
    [r0, r1] = w.receipts
    assert w.recall(r0.archive_keys[0]).id == "u0"
    assert [m.id for m in rendered[-2:]] == ["u2", "a3"]


def test_pinned_ids_skip_the_decider():
    decider = FakeDecider({"drop": 1.0, "pin": 0.0, "summarize": 0.0})
    w = SlidingWindow(decider, InMemoryArchive(), WindowPolicy(max_messages=2, pinned_ids=frozenset({"u0"})))
    w.add(_chat(4))
    assert w.receipts[0].reason == "pinned_by_user" and w.receipts[0].action == "pin"
    assert w.receipts[1].action == "drop"
    assert len(decider.calls) == 1 and [q.id for q in decider.calls[0]] == ["u1"]


@given(conversations(), cuts)
def test_decider_state_has_no_duplicates(msgs, cut):
    # Regression: evicted units were sent twice (as evicted and again as window).
    seen_states = []

    class Spy(FakeDecider):
        def decide(self, state, questions):
            seen_states.append(list(state))
            return super().decide(state, questions)

    w = SlidingWindow(Spy({"drop": 0.0, "pin": 1.0, "summarize": 0.0}), InMemoryArchive(), POLICY)
    feed(w, msgs, cut)
    for state in seen_states:
        ids = [m.id for m in state if m.id and not m.id.startswith("ctxsieve-note-")]
        assert len(ids) == len(set(ids))


def test_question_points_at_evicted_messages():
    decider = FakeDecider({"drop": 0.0, "pin": 1.0, "summarize": 0.0})
    w = SlidingWindow(decider, InMemoryArchive(), WindowPolicy(max_messages=2))
    w.add([
        HumanMessage("task", id="h0"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "search", "args": {}}], id="ai"),
        ToolMessage("result", tool_call_id="c1", id="t"),
        AIMessage("x", id="x"),
        HumanMessage("y", id="y"),
    ])
    [q] = decider.calls[0]
    # state = head (m0), evicted tool unit (m1, m2), window
    assert "Messages [m1] to [m2]" in q.text
    assert q.choices == ("drop", "pin", "summarize")
    assert isinstance(w.region[0], Pinned) and len(w.region[0].items) == 2
