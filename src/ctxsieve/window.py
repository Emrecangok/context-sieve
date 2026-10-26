"""Sliding window with decided eviction.

The last `max_messages` messages stay verbatim. When the window overflows, the oldest units
(a message, or an AI tool call together with its results) leave it, and one batched choice
question per unit decides what remains of them: `drop` (archive + pointer), `pin` (verbatim,
permanent) or `summarize` (queued verbatim until `summary.flush_pending` runs).
Messages inside the window are never changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .archive import Archive
from .deciders.base import Decider, Question, tag
from .receipts import EvictAction, EvictionReceipt
from .region import Dropped, Entry, Pending, Pinned, Seq, Summary

WINDOW_POLICY_VERSION = "0.1.0"
EVICT_CHOICES: tuple[EvictAction, ...] = ("drop", "pin", "summarize")
MARK = "ctxsieve"

log = logging.getLogger(__name__)

Unit = list[tuple[Seq, BaseMessage]]


@dataclass(frozen=True)
class WindowPolicy:
    max_messages: int = 20
    # Drop only when the decider is this sure; otherwise the unit is kept (pinned or queued).
    drop_min_prob: float = 0.8
    # Used when the decider fails or gives no usable answer. Both keep the text verbatim.
    fallback: Literal["pin", "summarize"] = "pin"
    pinned_ids: frozenset[str] = frozenset()


def choose_eviction(probs: Mapping[str, float] | None, policy: WindowPolicy) -> tuple[EvictAction, str]:
    if not probs or any(c not in probs for c in EVICT_CHOICES):
        return policy.fallback, "no_answer"
    if probs["drop"] >= policy.drop_min_prob:
        return "drop", "not_needed"
    if probs["pin"] >= probs["summarize"]:
        return "pin", "needs_verbatim"
    return "summarize", "gist_enough"


def eviction_question(qid: str, first: int, last: int) -> Question:
    span = f"Message {tag(first)}" if first == last else f"Messages {tag(first)} to {tag(last)}"
    return Question(
        qid,
        f"{span} are leaving the agent's recent-context window. Given the user's current task and "
        "what is likely to be needed later, choose what to keep of them. "
        "drop: not needed again. "
        "pin: contains a constraint, preference, identifier, decision or exact value that must stay verbatim. "
        "summarize: useful background where the gist is enough.",
        choices=EVICT_CHOICES,
    )


def default_note(text: str) -> BaseMessage:
    return HumanMessage(text, additional_kwargs={MARK: {"note": True}})


def _owns(unit: Unit, msg: ToolMessage) -> bool:
    first = unit[0][1]
    return isinstance(first, AIMessage) and any(tc.get("id") == msg.tool_call_id for tc in first.tool_calls)


class SlidingWindow:
    def __init__(self, decider: Decider, archive: Archive, policy: WindowPolicy = WindowPolicy()) -> None:
        self.decider = decider
        self.archive = archive
        self.policy = policy
        self.head: Unit = []
        self.region: list[Entry] = []
        self.window: list[Unit] = []
        self.receipts: list[EvictionReceipt] = []
        self.errors: list[str] = []
        self._seq = 0
        self._seen_human = False

    def add(self, messages: Sequence[BaseMessage]) -> list[EvictionReceipt]:
        for m in messages:
            self._append(m)
        return self._evict()

    def _append(self, m: BaseMessage) -> None:
        item = (self._seq, m)
        self._seq += 1
        if isinstance(m, SystemMessage) or (isinstance(m, HumanMessage) and not self._seen_human):
            self._seen_human |= isinstance(m, HumanMessage)
            self.head.append(item)
        elif isinstance(m, ToolMessage) and self.window and _owns(self.window[-1], m):
            self.window[-1].append(item)
        else:
            self.window.append([item])

    def _evict(self) -> list[EvictionReceipt]:
        size = sum(len(u) for u in self.window)
        evicted: list[Unit] = []
        # The newest unit always stays, even if it alone exceeds the limit.
        while size > self.policy.max_messages and len(self.window) > 1:
            unit = self.window.pop(0)
            size -= len(unit)
            evicted.append(unit)
        if not evicted:
            return []

        decisions = self._decide(evicted)
        receipts = []
        for unit, (action, reason, probs) in zip(evicted, decisions):
            keys: tuple[str, ...] = ()
            if action == "drop":
                try:
                    dropped = tuple((s, self.archive.put(m)) for s, m in unit)
                except Exception as e:  # keep the unit verbatim rather than lose it
                    log.warning("ctxsieve: archive failed, keeping unit instead of dropping: %r", e)
                    self.errors.append(repr(e))
                    action, reason = self.policy.fallback, "archive_error"
                else:
                    keys = tuple(k for _, k in dropped)
                    self.region.append(Dropped(dropped))
            if action == "pin":
                self.region.append(Pinned(tuple(unit)))
            elif action == "summarize":
                self.region.append(Pending(tuple(unit)))
            receipts.append(EvictionReceipt(
                seqs=tuple(s for s, _ in unit),
                message_ids=tuple(m.id for _, m in unit),
                action=action,
                reason=reason,
                probabilities=probs,
                decider=self.decider.name,
                policy_version=WINDOW_POLICY_VERSION,
                archive_keys=keys,
            ))
        self.receipts.extend(receipts)
        return receipts

    def _decide(self, evicted: list[Unit]) -> list[tuple[EvictAction, str, Mapping[str, float] | None]]:
        out: list[tuple[EvictAction, str, Mapping[str, float] | None] | None] = [None] * len(evicted)
        for i, unit in enumerate(evicted):
            if any(m.id is not None and m.id in self.policy.pinned_ids for _, m in unit):
                out[i] = ("pin", "pinned_by_user", None)
        ask = [i for i in range(len(evicted)) if out[i] is None]
        if ask:
            # State = what the agent currently sees, with the evicted units back in place.
            state = self._render_head_region(default_note)
            spans = []
            for unit in evicted:
                spans.append((len(state), len(state) + len(unit) - 1))
                state.extend(m for _, m in unit)
            state.extend(m for u in self.window for _, m in u)
            questions = [eviction_question(f"u{i}", *spans[i]) for i in ask]
            try:
                answers = {d.question_id: d.probs for d in self.decider.decide(state, questions)}
            except Exception as e:  # fail-open: keep everything verbatim
                log.warning("ctxsieve: decider %s failed during eviction: %r", self.decider.name, e)
                self.errors.append(repr(e))
                for i in ask:
                    out[i] = (self.policy.fallback, "decider_error", None)
            else:
                for i in ask:
                    probs = answers.get(f"u{i}")
                    out[i] = (*choose_eviction(probs, self.policy), probs)
        return out  # type: ignore[return-value]

    def _render_head_region(self, note: Callable[[str], BaseMessage]) -> list[BaseMessage]:
        out = [m for _, m in self.head]
        for e in self.region:
            if isinstance(e, (Pinned, Pending)):
                out.extend(m for _, m in e.items)
            elif isinstance(e, Dropped):
                keys = ", ".join(k for _, k in e.items)
                out.append(note(f"[ctxsieve: {len(e.items)} earlier message(s) dropped; archive {keys}]"))
            else:
                out.append(note(f"[ctxsieve: summary of {len(e.items)} earlier message(s)]\n{e.text}"))
        return out

    def render(self, note: Callable[[str], BaseMessage] = default_note) -> list[BaseMessage]:
        """The context to send to the model: head, evicted region, then the verbatim window."""
        return self._render_head_region(note) + [m for u in self.window for _, m in u]

    def recall(self, key: str) -> BaseMessage:
        return self.archive.get(key)

    def restore_all(self) -> list[BaseMessage]:
        """Every message ever added, in its original order."""
        items: list[tuple[Seq, BaseMessage]] = list(self.head)
        items.extend(p for u in self.window for p in u)
        for e in self.region:
            if isinstance(e, (Pinned, Pending)):
                items.extend(e.items)
            else:
                items.extend((s, self.archive.get(k)) for s, k in e.items)
        return [m for _, m in sorted(items, key=lambda p: p[0])]
