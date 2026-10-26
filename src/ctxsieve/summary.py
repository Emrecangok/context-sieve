"""Summary bucket for a SlidingWindow: flush queued units into append-only summary sections.

No summarizer ships with v0.1; plug one in through the `Summarizer` protocol. Pinned units
never enter a summary. A flush turns each contiguous run of queued units into a new section
in place and never rewrites earlier sections. Only `compact_summaries`, when the sections
outgrow `max_summary_chars`, merges adjacent sections of the same level into one section of
the next level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from langchain_core.messages import BaseMessage

from .region import Entry, Pending, Summary
from .window import SlidingWindow

log = logging.getLogger(__name__)


class Summarizer(Protocol):
    name: str

    def summarize(self, messages: Sequence[BaseMessage], previous: Sequence[str]) -> str:
        """Summarize `messages`. `previous` holds earlier sections for context only."""
        ...

    def merge(self, sections: Sequence[str]) -> str:
        """Condense several sections into one (hierarchical compaction)."""
        ...


@dataclass(frozen=True)
class SummaryPolicy:
    flush_at: int = 10  # queued messages needed before a flush
    max_summary_chars: int | None = None  # compaction trigger; None disables it


def pending_count(window: SlidingWindow) -> int:
    return sum(len(e.items) for e in window.region if isinstance(e, Pending))


def summary_chars(window: SlidingWindow) -> int:
    return sum(len(e.text) for e in window.region if isinstance(e, Summary))


def _runs(region: Sequence[Entry], same_run: Callable[[Entry, Entry | None], bool]) -> list[list[int]]:
    """Group adjacent indices; `same_run(entry, previous)` decides whether entry extends the run."""
    runs: list[list[int]] = []
    prev: Entry | None = None
    for i, e in enumerate(region):
        if runs and runs[-1][-1] == i - 1 and same_run(e, prev):
            runs[-1].append(i)
        elif same_run(e, None):
            runs.append([i])
        prev = e
    return runs


def _replace_runs(window: SlidingWindow, replaced: dict[int, tuple[list[int], Summary]]) -> None:
    skip = {i for run, _ in replaced.values() for i in run}
    out = []
    for i, e in enumerate(window.region):
        if i in replaced:
            out.append(replaced[i][1])
        elif i not in skip:
            out.append(e)
    window.region = out


def flush_pending(window: SlidingWindow, summarizer: Summarizer, policy: SummaryPolicy = SummaryPolicy()) -> list[Summary]:
    """Once at least `flush_at` messages are queued, summarize each contiguous run of them.

    Each run becomes one section in its own place, so nothing crosses a pinned or dropped
    entry. A run whose summarizer or archive call fails stays queued verbatim.
    """
    count = pending_count(window)
    if count == 0 or count < policy.flush_at:
        return []
    previous = [e.text for e in window.region if isinstance(e, Summary)]
    replaced: dict[int, tuple[list[int], Summary]] = {}
    for run in _runs(window.region, lambda e, _prev: isinstance(e, Pending)):
        items = [p for i in run for p in window.region[i].items]
        try:
            text = summarizer.summarize([m for _, m in items], previous)
            section = Summary(text, tuple((s, window.archive.put(m)) for s, m in items))
        except Exception as e:
            log.warning("ctxsieve: summary flush failed, run left queued: %r", e)
            window.errors.append(repr(e))
            continue
        replaced[run[0]] = (run, section)
        previous.append(text)
    _replace_runs(window, replaced)
    return [replaced[i][1] for i in sorted(replaced)]


def compact_summaries(window: SlidingWindow, summarizer: Summarizer, policy: SummaryPolicy) -> Summary | None:
    """When the summary text is over budget, merge the lowest-level, oldest run of adjacent
    same-level sections (at least two) into one section of the next level."""
    if policy.max_summary_chars is None or summary_chars(window) <= policy.max_summary_chars:
        return None

    def same_run(e: Entry, prev: Entry | None) -> bool:
        if not isinstance(e, Summary):
            return False
        return prev is None or (isinstance(prev, Summary) and prev.level == e.level)

    runs = [r for r in _runs(window.region, same_run) if len(r) >= 2]
    if not runs:
        return None
    run = min(runs, key=lambda r: (window.region[r[0]].level, r[0]))
    sections = [window.region[i] for i in run]
    try:
        text = summarizer.merge([s.text for s in sections])
    except Exception as e:
        log.warning("ctxsieve: summarizer %s failed to merge, sections left as is: %r", summarizer.name, e)
        window.errors.append(repr(e))
        return None
    merged = Summary(text, tuple(p for s in sections for p in s.items), level=sections[0].level + 1)
    _replace_runs(window, {run[0]: (run, merged)})
    return merged
