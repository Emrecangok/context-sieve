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
from typing import Any, Callable, Literal, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .archive import Archive
from .deciders.base import Decider, Question, tag, valid_prob
from .receipts import EvictAction, EvictionReceipt
from .region import Dropped, Entry, Pending, Pinned, Seq, Summary

WINDOW_POLICY_VERSION = "0.1.0"
STATE_VERSION = 1
EVICT_CHOICES: tuple[EvictAction, ...] = ("drop", "pin", "summarize")
MARK = "ctxsieve"
NOTE_ID_PREFIX = "ctxsieve-note-"

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
    if not isinstance(probs, Mapping) or not all(valid_prob(probs.get(c)) for c in EVICT_CHOICES):
        return policy.fallback, "no_answer"
    if probs["drop"] >= policy.drop_min_prob:
        return "drop", "not_needed"
    if probs["pin"] >= probs["summarize"]:
        return "pin", "needs_verbatim"
    return "summarize", "gist_enough"


def eviction_question(qid: str, first: int, last: int) -> Question:
    span = f"Message {tag(first)} is" if first == last else f"Messages {tag(first)} to {tag(last)} are"
    return Question(
        qid,
        f"{span} leaving the agent's recent-context window. Given the user's current task and "
        "what is likely to be needed later, choose what to keep of them. "
        "drop: not needed again. "
        "pin: contains a constraint, preference, identifier, decision or exact value that must stay verbatim. "
        "summarize: useful background where the gist is enough.",
        choices=EVICT_CHOICES,
    )


def default_note(text: str) -> BaseMessage:
    return HumanMessage(text, additional_kwargs={MARK: {"note": True}})


def is_note(m: BaseMessage) -> bool:
    """True for pointer/summary notes rendered by a SlidingWindow."""
    mark = m.additional_kwargs.get(MARK)
    return (isinstance(mark, dict) and bool(mark.get("note"))) or (m.id or "").startswith(NOTE_ID_PREFIX)


def _note(note: Callable[[str], BaseMessage], text: str, note_id: str, entry: dict[str, Any]) -> BaseMessage:
    # Stable ids, so a note rendered into graph state keeps its identity across turns. The
    # note also carries the region entry it stands for, so a window can be rebuilt from the
    # messages alone (see SlidingWindow.adopt).
    msg = note(text)
    kwargs = {**msg.additional_kwargs, MARK: {"note": True, "entry": entry}}
    return msg.model_copy(update={"id": msg.id or note_id, "additional_kwargs": kwargs})


MAX_KEYS_SHOWN = 5


def _dropped_text(items: Sequence[tuple[Seq, str]]) -> str:
    keys = [k for _, k in items]
    shown = ", ".join(keys[:MAX_KEYS_SHOWN]) + (f", +{len(keys) - MAX_KEYS_SHOWN} more" if len(keys) > MAX_KEYS_SHOWN else "")
    return f"[ctxsieve: {len(keys)} earlier message(s) dropped; archive {shown}]"


class MissingMessageError(KeyError):
    """A serialized window refers to a message that is no longer available."""


def _ids(items: Sequence[tuple[Seq, BaseMessage]]) -> list[list[Any]]:
    out = []
    for s, m in items:
        if not m.id:
            raise ValueError("SlidingWindow.to_state needs every message to have an id")
        out.append([s, m.id])
    return out


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
        self._first_human_seq: int | None = None

    def add(self, messages: Sequence[BaseMessage]) -> list[EvictionReceipt]:
        for m in messages:
            self._append(m)
        return self._evict()

    def _append(self, m: BaseMessage) -> None:
        item = (self._seq, m)
        self._seq += 1
        first_human = isinstance(m, HumanMessage) and not self._seen_human
        if first_human:
            self._seen_human = True
            self._first_human_seq = item[0]
        # Head is only the opening of the conversation, so rendering never reorders history.
        # A later system message or a first user message after a greeting stays in place and
        # is pinned automatically when it leaves the window.
        at_start = not self.window and not self.region
        if at_start and (isinstance(m, SystemMessage) or first_human):
            self.head.append(item)
        elif isinstance(m, ToolMessage) and self.window and _owns(self.window[-1], m):
            self.window[-1].append(item)
        else:
            self.window.append([item])

    def _evict(self) -> list[EvictionReceipt]:
        size = sum(len(u) for u in self.window)
        count = 0
        # The newest unit always stays, even if it alone exceeds the limit.
        while size > self.policy.max_messages and count < len(self.window) - 1:
            size -= len(self.window[count])
            count += 1
        if not count:
            return []

        # Decide before touching the window, so no failure can leave units in neither place.
        evicted = self.window[:count]
        try:
            decisions = self._decide(evicted)
        except Exception as e:
            log.warning("ctxsieve: eviction decision failed, keeping units verbatim: %r", e)
            self.errors.append(repr(e))
            decisions = [(self.policy.fallback, "decider_error", None)] * count
        del self.window[:count]
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
            elif any(isinstance(m, SystemMessage) for _, m in unit):
                out[i] = ("pin", "system_message", None)
            elif any(s == self._first_human_seq for s, _ in unit):
                out[i] = ("pin", "first_user_message", None)
        ask = [i for i in range(len(evicted)) if out[i] is None]
        if ask:
            # State = what the agent currently sees, with the evicted units back in place.
            state = self._render_head_region(default_note)
            spans = []
            for unit in evicted:
                spans.append((len(state), len(state) + len(unit) - 1))
                state.extend(m for _, m in unit)
            # `evicted` is still the head of self.window (it is removed after deciding).
            state.extend(m for u in self.window[len(evicted):] for _, m in u)
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
        run: list[tuple[Seq, str]] = []  # adjacent Dropped entries render as one pointer

        def flush_run() -> None:
            if run:
                entry = {"kind": "dropped", "items": [list(p) for p in run]}
                out.append(_note(note, _dropped_text(run), f"{NOTE_ID_PREFIX}dropped-{run[0][0]}", entry))
                run.clear()

        for e in self.region:
            if isinstance(e, Dropped):
                run.extend(e.items)
                continue
            flush_run()
            if isinstance(e, (Pinned, Pending)):
                out.extend(m for _, m in e.items)
            else:
                text = f"[ctxsieve: summary of {len(e.items)} earlier message(s)]\n{e.text}"
                entry = {"kind": "summary", "text": e.text, "level": e.level, "items": [list(p) for p in e.items]}
                out.append(_note(note, text, f"{NOTE_ID_PREFIX}summary-{e.items[0][0]}-{e.level}", entry))
        flush_run()
        return out

    def render(self, note: Callable[[str], BaseMessage] = default_note) -> list[BaseMessage]:
        """The context to send to the model: head, evicted region, then the verbatim window."""
        return self._render_head_region(note) + [m for u in self.window for _, m in u]

    def recall(self, key: str) -> BaseMessage:
        return self.archive.get(key)

    def known_ids(self) -> set[str]:
        """Ids of every message this window has consumed (archive keys included)."""
        ids = {m.id for _, m in self.head if m.id}
        ids.update(m.id for u in self.window for _, m in u if m.id)
        for e in self.region:
            if isinstance(e, (Pinned, Pending)):
                ids.update(m.id for _, m in e.items if m.id)
            else:
                ids.update(k for _, k in e.items)
        return ids

    def to_state(self) -> dict[str, Any]:
        """JSON-serializable state. Messages are stored by id, not copied."""
        region: list[dict[str, Any]] = []
        for e in self.region:
            if isinstance(e, Pinned):
                region.append({"kind": "pinned", "items": _ids(e.items)})
            elif isinstance(e, Pending):
                region.append({"kind": "pending", "items": _ids(e.items)})
            elif isinstance(e, Dropped):
                region.append({"kind": "dropped", "items": [list(p) for p in e.items]})
            else:
                region.append({"kind": "summary", "text": e.text, "level": e.level, "items": [list(p) for p in e.items]})
        return {
            "version": STATE_VERSION,
            "seq": self._seq,
            "seen_human": self._seen_human,
            "first_human_seq": self._first_human_seq,
            "head": _ids(self.head),
            "window": [_ids(u) for u in self.window],
            "region": region,
        }

    @classmethod
    def from_state(
        cls,
        data: Mapping[str, Any],
        messages: Sequence[BaseMessage],
        decider: Decider,
        archive: Archive,
        policy: WindowPolicy = WindowPolicy(),
    ) -> SlidingWindow:
        """Rebuild a window from `to_state()` output, resolving ids against `messages`.

        Raises MissingMessageError if a head, window, pinned or queued message is not in
        `messages` (e.g. another component rewrote the history).
        """
        if data.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported ctxsieve window state version: {data.get('version')!r}")
        by_id = {m.id: m for m in messages if m.id}

        def resolve(items: Sequence[Sequence[Any]]) -> tuple[tuple[Seq, BaseMessage], ...]:
            try:
                return tuple((int(s), by_id[i]) for s, i in items)
            except KeyError as e:
                raise MissingMessageError(e.args[0]) from None

        w = cls(decider, archive, policy)
        w._seq = int(data["seq"])
        w._seen_human = bool(data["seen_human"])
        first = data.get("first_human_seq")
        w._first_human_seq = None if first is None else int(first)
        w.head = list(resolve(data["head"]))
        w.window = [list(resolve(u)) for u in data["window"]]
        for e in data["region"]:
            kind = e["kind"]
            if kind == "pinned":
                w.region.append(Pinned(resolve(e["items"])))
            elif kind == "pending":
                w.region.append(Pending(resolve(e["items"])))
            elif kind == "dropped":
                w.region.append(Dropped(tuple((int(s), k) for s, k in e["items"])))
            elif kind == "summary":
                w.region.append(Summary(e["text"], tuple((int(s), k) for s, k in e["items"]), int(e["level"])))
            else:
                raise ValueError(f"unknown region entry kind: {kind!r}")
        return w

    @classmethod
    def adopt(
        cls,
        messages: Sequence[BaseMessage],
        decider: Decider,
        archive: Archive,
        policy: WindowPolicy = WindowPolicy(),
    ) -> SlidingWindow:
        """Rebuild a window from an already-pruned message list, losing nothing.

        Used when stored window state is missing or unusable but the messages may already
        contain notes (state mode). Notes become their Dropped/Summary entries again, every
        other message older than the window is kept verbatim (pinned), and no decider call is
        made. Rendering the result reproduces `messages` in order.
        """
        w = cls(decider, archive, policy)
        units: list[tuple[str, Any]] = []  # ("note", entry) or ("unit", Unit)
        seq = 0
        for m in messages:
            mark = m.additional_kwargs.get(MARK)
            entry = mark.get("entry") if isinstance(mark, dict) else None
            if is_note(m):
                if isinstance(entry, dict) and entry.get("kind") in ("dropped", "summary"):
                    items = tuple((int(s), str(k)) for s, k in entry.get("items", []))
                    if items:
                        seq = max(seq, max(s for s, _ in items) + 1)
                        made: Entry = (
                            Dropped(items) if entry["kind"] == "dropped"
                            else Summary(str(entry.get("text", "")), items, int(entry.get("level", 0)))
                        )
                        units.append(("note", made))
                        continue
                # A note without usable metadata is kept as a plain message.
            item = (seq, m)
            seq += 1
            if isinstance(m, ToolMessage) and units and units[-1][0] == "unit" and _owns(units[-1][1], m):
                units[-1][1].append(item)
            else:
                units.append(("unit", [item]))

        # The newest units that fit stay in the window; everything before is region.
        size, split = 0, len(units)
        while split > 0 and units[split - 1][0] == "unit":
            unit_len = len(units[split - 1][1])
            if size + unit_len > policy.max_messages and split < len(units):
                break
            size += unit_len
            split -= 1
        for kind, value in units[:split]:
            w.region.append(value if kind == "note" else Pinned(tuple(value)))
        w.window = [value for _, value in units[split:]]
        w._seq = seq
        for _, value in units:
            if isinstance(value, list):
                for s, m in value:
                    if isinstance(m, HumanMessage) and not is_note(m):
                        w._seen_human = True
                        w._first_human_seq = s
                        break
            if w._seen_human:
                break
        return w

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
