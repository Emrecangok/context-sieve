from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from conftest import FakeSummarizer, chunks, conversations
from ctxsieve import InMemoryArchive, SlidingWindow, SummaryPolicy, WindowPolicy, compact_summaries, flush_pending
from ctxsieve.deciders import FakeDecider
from ctxsieve.region import Pending, Pinned, Summary
from ctxsieve.summary import pending_count

SUMMARIZE = {"drop": 0.0, "pin": 0.0, "summarize": 1.0}


def _chat(n):
    msgs = [SystemMessage("sys", id="s"), HumanMessage("task", id="h0")]
    for i in range(n):
        msgs.append(AIMessage(f"a{i}", id=f"a{i}") if i % 2 else HumanMessage(f"u{i}", id=f"u{i}"))
    return msgs


def _feed_one_by_one(w, msgs, after=lambda: None):
    w.add(msgs[:2])
    for m in msgs[2:]:
        w.add([m])
        after()


def test_no_flush_below_threshold():
    w = SlidingWindow(FakeDecider(SUMMARIZE), InMemoryArchive(), WindowPolicy(max_messages=2))
    w.add(_chat(5))  # evicts u0, a1, u2
    s = FakeSummarizer()
    assert flush_pending(w, s, SummaryPolicy(flush_at=4)) == []
    assert s.calls == [] and pending_count(w) == 3


def test_flush_replaces_queue_with_one_section():
    w = SlidingWindow(FakeDecider(SUMMARIZE), InMemoryArchive(), WindowPolicy(max_messages=2))
    msgs = _chat(6)
    w.add(msgs)
    [section] = flush_pending(w, FakeSummarizer(), SummaryPolicy(flush_at=4))
    assert section.text == "S(u0,a1,u2,a3)"
    assert w.region == [section]
    assert w.restore_all() == msgs
    assert "S(u0,a1,u2,a3)" in w.render()[2].text


def test_flush_never_moves_content_across_a_pin():
    # Review finding: a later "budget now 200" must not be summarized ahead of the pinned "budget 100".
    w = SlidingWindow(FakeDecider(SUMMARIZE), InMemoryArchive(), WindowPolicy(max_messages=1, pinned_ids=frozenset({"b"})))
    w.add([
        SystemMessage("sys", id="s"),
        HumanMessage("task", id="h"),
        HumanMessage("use postgres", id="a"),
        HumanMessage("constraint: budget 100", id="b"),
        HumanMessage("actually budget now 200", id="c"),
        AIMessage("ok", id="d"),
    ])
    sections = flush_pending(w, FakeSummarizer(), SummaryPolicy(flush_at=2))
    assert [s.text for s in sections] == ["S(a)", "S(c)"]
    assert [type(e) for e in w.region] == [Summary, Pinned, Summary]
    texts = [m.text for m in w.render()]
    assert texts.index("constraint: budget 100") < next(i for i, t in enumerate(texts) if "S(c)" in t)


def test_sections_are_append_only_and_pins_stay_out():
    w = SlidingWindow(
        FakeDecider(SUMMARIZE), InMemoryArchive(), WindowPolicy(max_messages=2, pinned_ids=frozenset({"a1", "u4"}))
    )
    msgs = _chat(10)
    summarizer = FakeSummarizer()
    sections = []

    def flush():
        new = flush_pending(w, summarizer, SummaryPolicy(flush_at=2))
        for s in sections:  # earlier sections are never rewritten
            assert s in w.region
        sections.extend(new)

    _feed_one_by_one(w, msgs, flush)
    assert len(sections) >= 2
    pinned_ids = {m.id for e in w.region if isinstance(e, Pinned) for _, m in e.items}
    summarized = {i for c in summarizer.calls for i in c[1]}
    assert pinned_ids == {"a1", "u4"} and not (pinned_ids & summarized)
    # Earlier sections are passed for context.
    assert set(summarizer.calls[-1][2]) >= {s.text for s in sections[:-1]}
    assert w.restore_all() == msgs


def test_summarizer_failure_leaves_queue():
    w = SlidingWindow(FakeDecider(SUMMARIZE), InMemoryArchive(), WindowPolicy(max_messages=2))
    w.add(_chat(6))
    before = list(w.region)
    assert flush_pending(w, FakeSummarizer(error=RuntimeError("x")), SummaryPolicy(flush_at=1)) == []
    assert w.region == before and all(isinstance(e, Pending) for e in w.region)


def test_archive_failure_during_flush_leaves_queue():
    class BrokenArchive(InMemoryArchive):
        fail = False

        def put(self, message):
            if self.fail:
                raise OSError("disk full")
            return super().put(message)

    archive = BrokenArchive()
    w = SlidingWindow(FakeDecider(SUMMARIZE), archive, WindowPolicy(max_messages=2))
    msgs = _chat(6)
    w.add(msgs)
    archive.fail = True
    assert flush_pending(w, FakeSummarizer(), SummaryPolicy(flush_at=1)) == []
    assert all(isinstance(e, Pending) for e in w.region)
    assert w.restore_all() == msgs


def test_compaction_only_when_over_limit():
    w = SlidingWindow(FakeDecider(SUMMARIZE), InMemoryArchive(), WindowPolicy(max_messages=2))
    msgs = _chat(12)
    s = FakeSummarizer()
    _feed_one_by_one(w, msgs, lambda: flush_pending(w, s, SummaryPolicy(flush_at=2)))
    n_sections = sum(isinstance(e, Summary) for e in w.region)
    assert n_sections >= 3

    assert compact_summaries(w, s, SummaryPolicy(max_summary_chars=10_000)) is None
    assert compact_summaries(w, s, SummaryPolicy(max_summary_chars=None)) is None

    merged = compact_summaries(w, s, SummaryPolicy(max_summary_chars=10))
    assert merged is not None and merged.level == 1 and merged.text.startswith("M(")
    assert sum(isinstance(e, Summary) for e in w.region) < n_sections
    assert w.restore_all() == msgs


def _depth(text: str) -> int:
    d = 0
    while text.startswith("M(" * (d + 1)):
        d += 1
    return d


def test_compaction_is_hierarchical_not_a_chain():
    # Review finding: merging the previous merge output with each new section built M(M(M(...))).
    w = SlidingWindow(FakeDecider(SUMMARIZE), InMemoryArchive(), WindowPolicy(max_messages=2))
    msgs = _chat(34)
    s = FakeSummarizer()

    def step():
        flush_pending(w, s, SummaryPolicy(flush_at=2))
        compact_summaries(w, s, SummaryPolicy(max_summary_chars=30))

    _feed_one_by_one(w, msgs, step)
    merges = [c[1] for c in s.calls if c[0] == "merge"]
    assert merges
    for sections in merges:
        assert len({_depth(t) for t in sections}) == 1  # only same-level sections are merged
    assert max(e.level for e in w.region if isinstance(e, Summary)) <= 3
    assert w.restore_all() == msgs


@given(conversations(), st.lists(st.integers(0, 60), max_size=5), st.integers(1, 6), st.integers(5, 60))
def test_restore_survives_flush_and_compaction(msgs, cuts, flush_at, max_chars):
    w = SlidingWindow(FakeDecider(SUMMARIZE), InMemoryArchive(), WindowPolicy(max_messages=3))
    s = FakeSummarizer()
    for chunk in chunks(msgs, cuts):
        w.add(chunk)
        flush_pending(w, s, SummaryPolicy(flush_at=flush_at))
        compact_summaries(w, s, SummaryPolicy(max_summary_chars=max_chars))
    assert w.restore_all() == msgs
