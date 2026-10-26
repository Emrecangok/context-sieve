from __future__ import annotations

from typing import Callable, Sequence

from langchain_core.messages import BaseMessage

from .base import Decision, Question


class FakeDecider:
    """Offline decider for tests: a constant, or a function of the question."""

    name = "fake"

    def __init__(
        self,
        p_yes: float | None | Callable[[Question], float | None] = 1.0,
        error: Exception | None = None,
    ) -> None:
        self._p_yes = p_yes
        self._error = error
        self.calls: list[list[Question]] = []

    def decide(self, state: Sequence[BaseMessage], questions: Sequence[Question]) -> list[Decision]:
        self.calls.append(list(questions))
        if self._error is not None:
            raise self._error
        fn = self._p_yes if callable(self._p_yes) else (lambda _q: self._p_yes)
        return [Decision(q.id, fn(q)) for q in questions]
