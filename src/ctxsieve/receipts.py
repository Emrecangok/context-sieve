from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

Action = Literal["keep", "truncate", "clear"]


@dataclass(frozen=True)
class Receipt:
    """One pruning decision: what was asked, what came back, what was done."""

    index: int
    message_id: str | None
    tool_call_id: str | None
    action: Action
    reason: str
    probabilities: Mapping[str, float | None]
    decider: str
    policy_version: str
    chars_before: int
    chars_after: int
    archive_key: str | None = None
    raw: Mapping[str, object] = field(default_factory=dict)


EvictAction = Literal["drop", "pin", "summarize"]


@dataclass(frozen=True)
class EvictionReceipt:
    """One eviction decision for a unit (a message, or a tool call with its results)."""

    seqs: tuple[int, ...]
    message_ids: tuple[str | None, ...]
    action: EvictAction
    reason: str
    probabilities: Mapping[str, float] | None
    decider: str
    policy_version: str
    archive_keys: tuple[str, ...] = ()
