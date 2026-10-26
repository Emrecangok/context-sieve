"""Pure pruning core: decide which tool results to keep, truncate or clear.

v0.1 only edits ToolMessage content. AI messages (and their tool calls) are never
touched, so every call keeps its result message and the pair stays valid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .archive import Archive
from .deciders.base import Decider, Decision, Question, tag, valid_prob
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


def _qid(index: int, aspect: str) -> str:
    # By position: tool_call_ids are not guaranteed unique (some providers reuse them).
    return f"m{index}:{aspect}"


def cache_key(msg: BaseMessage) -> str | None:
    """Key for a cached decision: the message id, else its tool_call_id."""
    return msg.id or getattr(msg, "tool_call_id", None)


def questions_for(index: int, msg: ToolMessage) -> list[Question]:
    ref = f"tool call {msg.tool_call_id} ({msg.name or 'tool'}), output message {tag(index)}"
    return [
        Question(
            _qid(index, "verbatim"),
            f"The full, exact output of {ref} is still needed to complete the user's current task.",
        ),
        Question(
            _qid(index, "gist"),
            f"Knowing that {ref} was made, plus a short excerpt of its output, "
            "is still useful for the user's current task.",
        ),
    ]


def _p(decisions: dict[str, Decision], qid: str) -> float | None:
    d = decisions.get(qid)
    return d.p_yes if d is not None and valid_prob(d.p_yes) else None


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


# Reasons that reflect a real answer and may be reused instead of asking again.
CACHEABLE_REASONS = frozenset({"may_need_verbatim", "gist_only", "not_needed", "too_short_to_truncate"})


def prune(
    messages: Sequence[BaseMessage],
    decider: Decider,
    archive: Archive,
    policy: PrunePolicy = PrunePolicy(),
    decided: Mapping[str, Action] | None = None,
) -> PruneResult:
    """Keep, truncate or clear older tool results.

    `decided` maps cache_key(message) -> action from earlier calls; those results are not
    asked about again and their receipt has reason "cached".
    """
    messages = list(messages)
    decided = {k: v for k, v in (decided or {}).items() if v in ("keep", "truncate", "clear")}
    candidates = candidate_indices(messages, policy)
    if not candidates:
        return PruneResult(messages)

    ask = [i for i in candidates if cache_key(messages[i]) not in decided]
    decisions: dict[str, Decision] = {}
    error = None
    if ask:
        questions = [q for i in ask for q in questions_for(i, messages[i])]
        try:
            decisions = {d.question_id: d for d in decider.decide(messages, questions)}
        except Exception as e:  # fail-open: never prune on decider failure
            log.warning("ctxsieve: decider %s failed, leaving messages untouched: %r", decider.name, e)
            error = repr(e)

    out = list(messages)
    receipts = []
    for i in candidates:
        msg = messages[i]
        probs: dict[str, float | None] = {}
        if cache_key(msg) in decided:
            action, reason = decided[cache_key(msg)], "cached"
        elif error is not None:
            action, reason = "keep", "decider_error"
        else:
            probs = {aspect: _p(decisions, _qid(i, aspect)) for aspect in ("verbatim", "gist")}
            action, reason = choose_action(probs["verbatim"], probs["gist"], policy)
        if action == "truncate" and len(msg.text) <= policy.truncate_head_chars:
            action, reason = "keep", "too_short_to_truncate"
        key = None
        if action != "keep":
            try:
                key = archive.put(msg)
            except Exception as e:  # fail-open: keep the original rather than lose it
                log.warning("ctxsieve: archive failed, keeping tool result: %r", e)
                error = error or repr(e)
                action, reason = "keep", "archive_error"
            else:
                out[i] = _replace(msg, action, key, policy)
        receipts.append(_receipt(i, msg, action, reason, probs, decider, key, out[i]))
    return PruneResult(out, receipts, error=error)


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
