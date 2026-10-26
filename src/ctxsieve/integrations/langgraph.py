"""LangGraph integration: hooks usable from any StateGraph node.

Two modes, chosen per hook:

- "view": graph state keeps the full history untouched. `update()` only records decisions
  under its own state key, and `view()` returns what the model should see. Nothing is ever
  removed, so the history itself is the archive (message ids are the archive keys).
- "state": `update()` rewrites the `messages` channel to the pruned context. Removed
  originals go to an Archive, by default a `StoreArchive` on the graph's store, so they
  survive restarts and can be restored.

Typical wiring (view mode):

    hook = SlidingWindowHook(decider, WindowPolicy(max_messages=20))

    def sieve(state):            # node before the model node
        return hook.update(state)

    def call_model(state):
        return {"messages": [llm.invoke(hook.view(state))]}

The graph state needs the hook's key (see `CtxSieveState`). In state mode pass the
store: `hook.update(state, store=runtime.store)`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Mapping, Sequence

from langchain_core.messages import BaseMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing_extensions import NotRequired, TypedDict

from ..archive import Archive, MessageIdArchive, StoreArchive
from ..deciders.base import Decider, Decision, Question
from ..policy import CACHEABLE_REASONS, PrunePolicy, prune
from ..receipts import EvictionReceipt, Receipt
from ..summary import Summarizer, SummaryPolicy, compact_summaries, flush_pending
from ..window import MissingMessageError, SlidingWindow, WindowPolicy, default_note, is_note

WINDOW_KEY = "ctxsieve_window"
PRUNE_KEY = "ctxsieve_prune"

Mode = Literal["view", "state"]

log = logging.getLogger(__name__)


class CtxSieveState(TypedDict, total=False):
    """Keys the hooks keep in graph state. Merge into your own state schema."""

    ctxsieve_window: dict[str, Any]
    ctxsieve_prune: dict[str, Any]


def _check_mode(mode: str) -> Mode:
    if mode not in ("view", "state"):
        raise ValueError(f"mode must be 'view' or 'state', got {mode!r}")
    return mode  # type: ignore[return-value]


def _ids_ok(messages: Sequence[BaseMessage]) -> bool:
    """Hooks need stable message ids (the `add_messages` reducer assigns them)."""
    if all(m.id for m in messages if not is_note(m)):
        return True
    log.warning("ctxsieve: messages without ids; use the add_messages reducer. Skipping this turn.")
    return False


def _same(a: Sequence[BaseMessage], b: Sequence[BaseMessage]) -> bool:
    return len(a) == len(b) and all(x.id == y.id and x.content == y.content for x, y in zip(a, b))


class _NoAnswers:
    """Decider that answers nothing: used to render without asking new questions."""

    name = "none"

    def decide(self, state: Sequence[BaseMessage], questions: Sequence[Question]) -> list[Decision]:
        return []


class SlidingWindowHook:
    def __init__(
        self,
        decider: Decider,
        policy: WindowPolicy = WindowPolicy(),
        *,
        mode: Mode = "view",
        archive: Archive | None = None,
        summarizer: Summarizer | None = None,
        summary_policy: SummaryPolicy = SummaryPolicy(),
        note: Callable[[str], BaseMessage] = default_note,
        on_receipts: Callable[[list[EvictionReceipt]], None] | None = None,
    ) -> None:
        self.decider = decider
        self.policy = policy
        self.mode = _check_mode(mode)
        self.archive = archive
        self.summarizer = summarizer
        self.summary_policy = summary_policy
        self.note = note
        self.on_receipts = on_receipts

    def _archive(self, messages: Sequence[BaseMessage], store: Any) -> Archive:
        if self.mode == "view":
            return MessageIdArchive(messages)
        if self.archive is not None:
            return self.archive
        if store is not None:
            return StoreArchive(store)
        raise ValueError("state mode needs an archive: pass archive=... or the graph's store")

    def _load(self, state: Mapping[str, Any], messages: Sequence[BaseMessage], archive: Archive) -> SlidingWindow:
        data = state.get(WINDOW_KEY)
        broken = False
        if data:
            try:
                return SlidingWindow.from_state(data, messages, self.decider, archive, self.policy)
            except (MissingMessageError, ValueError, KeyError, TypeError, AttributeError) as e:
                log.warning("ctxsieve: window state unusable (%r); rebuilding from current messages", e)
                broken = True
        if self.mode == "state" and (broken or any(is_note(m) for m in messages)):
            # The channel may already be pruned: rebuild from it as is, so no note is lost.
            return SlidingWindow.adopt(messages, self.decider, archive, self.policy)
        return SlidingWindow(self.decider, archive, self.policy)

    def update(self, state: Mapping[str, Any], store: Any = None) -> dict[str, Any]:
        """Consume new messages, decide evictions, and return the state update for a node."""
        messages = list(state["messages"])
        if not _ids_ok(messages):
            return {}
        archive = self._archive(messages, store)
        window = self._load(state, messages, archive)
        known = window.known_ids()
        new = [m for m in messages if m.id not in known and not is_note(m)]
        receipts = window.add(new)
        if self.summarizer is not None:
            flush_pending(window, self.summarizer, self.summary_policy)
            compact_summaries(window, self.summarizer, self.summary_policy)
        if receipts:
            (self.on_receipts or _log_receipts)(receipts)

        out: dict[str, Any] = {WINDOW_KEY: window.to_state()}
        if self.mode == "state":
            rendered = window.render(self.note)
            if not _same(rendered, messages):
                out["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *rendered]
        return out

    def view(self, state: Mapping[str, Any], messages: Sequence[BaseMessage] | None = None) -> list[BaseMessage]:
        """What the model should see. In state mode the state already is the view.

        Messages that arrived after the last `update()` are appended verbatim; nothing is
        asked or evicted here.
        """
        messages = list(state["messages"] if messages is None else messages)
        if self.mode == "state" or not state.get(WINDOW_KEY):
            return messages
        window = self._load(state, messages, MessageIdArchive(messages))
        known = window.known_ids()
        # Messages the window does not know keep their place relative to the window's span:
        # before its first known message, or after it.
        rendered = window.render(self.note)
        out: list[BaseMessage] = []
        placed = False
        for m in messages:
            if m.id is not None and m.id in known:
                if not placed:
                    out.extend(rendered)
                    placed = True
            else:
                out.append(m)
        return out if placed else rendered + out

    def restore(self, state: Mapping[str, Any], store: Any = None) -> list[BaseMessage]:
        """The full original history (state mode: pulled back from the archive)."""
        messages = list(state["messages"])
        if self.mode == "view":
            return [m for m in messages if not is_note(m)]
        window = self._load(state, messages, self._archive(messages, store))
        known = window.known_ids()
        tail = [m for m in messages if m.id not in known and not is_note(m)]
        return window.restore_all() + tail


class ToolPruneHook:
    def __init__(
        self,
        decider: Decider,
        policy: PrunePolicy = PrunePolicy(),
        *,
        mode: Mode = "view",
        archive: Archive | None = None,
        on_receipts: Callable[[list[Receipt]], None] | None = None,
    ) -> None:
        self.decider = decider
        self.policy = policy
        self.mode = _check_mode(mode)
        self.archive = archive
        self.on_receipts = on_receipts

    def _archive(self, messages: Sequence[BaseMessage], store: Any) -> Archive:
        if self.mode == "view":
            return MessageIdArchive(messages)
        if self.archive is not None:
            return self.archive
        if store is not None:
            return StoreArchive(store)
        raise ValueError("state mode needs an archive: pass archive=... or the graph's store")

    @staticmethod
    def _decided(state: Mapping[str, Any]) -> dict[str, str]:
        data = state.get(PRUNE_KEY) or {}
        decided = data.get("decided") if isinstance(data, Mapping) else None
        return dict(decided) if isinstance(decided, Mapping) else {}

    def update(self, state: Mapping[str, Any], store: Any = None) -> dict[str, Any]:
        messages = list(state["messages"])
        if not _ids_ok(messages):
            return {}
        decided = self._decided(state)
        res = prune(messages, self.decider, self._archive(messages, store), self.policy, decided=decided)
        fresh = [r for r in res.receipts if r.reason != "cached"]
        for r in fresh:
            key = r.message_id or r.tool_call_id
            if r.reason in CACHEABLE_REASONS and key:
                decided[key] = r.action
        if fresh:
            (self.on_receipts or _log_receipts)(fresh)

        out: dict[str, Any] = {PRUNE_KEY: {"decided": decided}}
        if self.mode == "state":
            changed = [new for new, old in zip(res.messages, messages) if new is not old]
            if changed:
                out["messages"] = changed  # same ids: add_messages replaces them in place
        return out

    def view(self, state: Mapping[str, Any], messages: Sequence[BaseMessage] | None = None) -> list[BaseMessage]:
        messages = list(state["messages"] if messages is None else messages)
        if self.mode == "state" or not _ids_ok(messages):
            return messages
        res = prune(messages, _NoAnswers(), MessageIdArchive(messages), self.policy, decided=self._decided(state))
        return res.messages


def _log_receipts(receipts: Sequence[Any]) -> None:
    for r in receipts:
        log.debug("ctxsieve receipt: %s", r)
