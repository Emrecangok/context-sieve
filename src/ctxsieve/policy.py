"""Pure pruning core: decide which tool results to keep, truncate or clear.

v0.1 only edits ToolMessage content. AI messages (and their tool calls) are never
touched, so every call keeps its result message and the pair stays valid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .archive import Archive
from .deciders.base import Decider, Decision, Question, tag
from .receipts import Action, Receipt

POLICY_VERSION = "0.1.0"
MARK = "ctxsieve"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrunePolicy:
    # Act only when the decider is confident the content is NOT needed (default: keep).
    act_below: float = 0.2
    preserve_recent: int = 6
    truncate_head_chars: int = 300
    pinned_ids: frozenset[str] = frozenset()


@dataclass
class PruneResult:
    messages: list[BaseMessage]
    receipts: list[Receipt] = field(default_factory=list)
    error: str | None = None


def protected_indices(messages: Sequence[BaseMessage], policy: PrunePolicy) -> set[int]:
    n = len(messages)
    protected = set(range(max(0, n - policy.preserve_recent), n))
    first_human = next((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), None)
    if first_human is not None:
        protected.add(first_human)
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage) or (m.id is not None and m.id in policy.pinned_ids):
            protected.add(i)
    return protected


def candidate_indices(messages: Sequence[BaseMessage], policy: PrunePolicy) -> list[int]:
    protected = protected_indices(messages, policy)
    seen_calls: set[str] = set()
    out = []
    for i, m in enumerate(messages):
        if isinstance(m, AIMessage):
            seen_calls.update(tc["id"] for tc in m.tool_calls if tc.get("id"))
        elif (
            isinstance(m, ToolMessage)
            and i not in protected
            and MARK not in m.additional_kwargs
            and m.tool_call_id in seen_calls  # leave orphan results alone
        ):
            out.append(i)
    return out


def _qid(msg: ToolMessage, aspect: str) -> str:
    return f"{msg.tool_call_id}:{aspect}"


def questions_for(index: int, msg: ToolMessage) -> list[Question]:
    ref = f"tool call {msg.tool_call_id} ({msg.name or 'tool'}), output message {tag(index)}"
    return [
        Question(
            _qid(msg, "verbatim"),
            f"The full, exact output of {ref} is still needed to complete the user's current task.",
        ),
        Question(
            _qid(msg, "gist"),
            f"Knowing that {ref} was made, plus a short excerpt of its output, "
            "is still useful for the user's current task.",
        ),
    ]


def _p(decisions: dict[str, Decision], qid: str) -> float | None:
    d = decisions.get(qid)
    return None if d is None else d.p_yes


def choose_action(p_verbatim: float | None, p_gist: float | None, policy: PrunePolicy) -> tuple[Action, str]:
    if p_verbatim is None:
        return "keep", "no_answer"
    if p_verbatim >= policy.act_below:
        return "keep", "may_need_verbatim"
    if p_gist is None:
        return "keep", "no_answer"
    if p_gist >= policy.act_below:
        return "truncate", "gist_only"
    return "clear", "not_needed"


def _replace(msg: ToolMessage, action: Action, key: str, policy: PrunePolicy) -> ToolMessage:
    text = msg.text
    if action == "truncate":
        body = f"{text[:policy.truncate_head_chars]}\n[ctxsieve: truncated from {len(text)} chars, archive {key}]"
    else:
        body = f"[ctxsieve: cleared {len(text)} chars, archive {key}]"
    kwargs = {**msg.additional_kwargs, MARK: {"action": action, "archive_key": key}}
    return msg.model_copy(update={"content": body, "additional_kwargs": kwargs})


def prune(
    messages: Sequence[BaseMessage],
    decider: Decider,
    archive: Archive,
    policy: PrunePolicy = PrunePolicy(),
) -> PruneResult:
    messages = list(messages)
    candidates = candidate_indices(messages, policy)
    if not candidates:
        return PruneResult(messages)

    questions = [q for i in candidates for q in questions_for(i, messages[i])]
    try:
        decisions = {d.question_id: d for d in decider.decide(messages, questions)}
    except Exception as e:  # fail-open: never prune on decider failure
        log.warning("ctxsieve: decider %s failed, leaving messages untouched: %r", decider.name, e)
        receipts = [_receipt(i, messages[i], "keep", "decider_error", {}, decider, None, messages[i]) for i in candidates]
        return PruneResult(messages, receipts, error=repr(e))

    out = list(messages)
    receipts = []
    for i in candidates:
        msg = messages[i]
        probs = {aspect: _p(decisions, _qid(msg, aspect)) for aspect in ("verbatim", "gist")}
        action, reason = choose_action(probs["verbatim"], probs["gist"], policy)
        if action == "truncate" and len(msg.text) <= policy.truncate_head_chars:
            action, reason = "keep", "too_short_to_truncate"
        key = None
        if action != "keep":
            key = archive.put(msg)
            out[i] = _replace(msg, action, key, policy)
        receipts.append(_receipt(i, msg, action, reason, probs, decider, key, out[i]))
    return PruneResult(out, receipts)


def _receipt(
    index: int,
    before: ToolMessage,
    action: Action,
    reason: str,
    probs: dict[str, float | None],
    decider: Decider,
    key: str | None,
    after: BaseMessage,
) -> Receipt:
    return Receipt(
        index=index,
        message_id=before.id,
        tool_call_id=before.tool_call_id,
        action=action,
        reason=reason,
        probabilities=probs,
        decider=decider.name,
        policy_version=POLICY_VERSION,
        chars_before=len(before.text),
        chars_after=len(after.text),
        archive_key=key,
    )


def restore(messages: Sequence[BaseMessage], archive: Archive) -> list[BaseMessage]:
    """Undo pruning: swap every marked message back for its archived original."""
    return [
        archive.get(m.additional_kwargs[MARK]["archive_key"]) if MARK in m.additional_kwargs else m
        for m in messages
    ]
