from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class Question:
    """A yes/no (predicate) question about the conversation state."""

    id: str
    text: str


@dataclass(frozen=True)
class Decision:
    question_id: str
    # None means the backend gave no usable answer (refusal, missing, gated); policy keeps.
    p_yes: float | None
    raw: Mapping[str, Any] = field(default_factory=dict)


class Decider(Protocol):
    name: str

    def decide(self, state: Sequence[BaseMessage], questions: Sequence[Question]) -> list[Decision]: ...
