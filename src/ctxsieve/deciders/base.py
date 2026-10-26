from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from langchain_core.messages import BaseMessage


def valid_prob(p: Any) -> bool:
    """A usable probability: a finite real number in [0, 1]. Anything else counts as no answer."""
    return isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0.0 <= p <= 1.0


def tag(index: int) -> str:
    """Label prepended to message `index` of the state so questions can refer to it."""
    return f"[m{index}]"


@dataclass(frozen=True)
class Question:
    """A predicate (yes/no) question, or a choice question when `choices` is set."""

    id: str
    text: str
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Decision:
    question_id: str
    # Predicate answer. None means no usable answer (refusal, missing, gated); policy keeps.
    p_yes: float | None = None
    # Choice answer: label -> probability. None means no usable answer.
    probs: Mapping[str, float] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


class Decider(Protocol):
    name: str

    def decide(self, state: Sequence[BaseMessage], questions: Sequence[Question]) -> list[Decision]: ...
