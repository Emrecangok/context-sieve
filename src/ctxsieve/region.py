"""Entries of the evicted region: what remains of messages that left the sliding window.

The region is always in chronological order. Evictions only append to it, so the rendered
prefix stays stable between evictions. A summary flush replaces a contiguous run of Pending
entries in place, and compaction merges a contiguous run of same-level Summary entries, so
neither moves content across a Pinned or Dropped entry. Each message carries its sequence
number, which lets the full history be restored in its original order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from langchain_core.messages import BaseMessage

Seq = int


@dataclass(frozen=True)
class Pinned:
    """Kept verbatim and permanently."""

    items: tuple[tuple[Seq, BaseMessage], ...]


@dataclass(frozen=True)
class Dropped:
    """Removed from context; originals live in the archive."""

    items: tuple[tuple[Seq, str], ...]  # (seq, archive key)


@dataclass(frozen=True)
class Pending:
    """Marked for summary; stays verbatim until a summarizer flushes it."""

    items: tuple[tuple[Seq, BaseMessage], ...]


@dataclass(frozen=True)
class Summary:
    """One append-only summary section; originals live in the archive."""

    text: str
    items: tuple[tuple[Seq, str], ...]  # (seq, archive key) of every covered message
    level: int = 0


Entry = Union[Pinned, Dropped, Pending, Summary]
