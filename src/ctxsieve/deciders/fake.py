from __future__ import annotations

from typing import Callable, Mapping, Sequence, Union

from langchain_core.messages import BaseMessage

from .base import Decision, Question

Answer = Union[float, Mapping[str, float], None]


class FakeDecider:
    """Offline decider for tests: a constant, or a function of the question.

    Predicate questions take a float; choice questions take a label -> probability mapping.
    An answer of the wrong shape counts as no answer.
    """

    name = "fake"

    def __init__(
        self,
        answer: Answer | Callable[[Question], Answer] = 1.0,
        error: Exception | None = None,
    ) -> None:
        self._answer = answer
        self._error = error
        self.calls: list[list[Question]] = []

    def decide(self, state: Sequence[BaseMessage], questions: Sequence[Question]) -> list[Decision]:
        self.calls.append(list(questions))
        if self._error is not None:
            raise self._error
        fn = self._answer if callable(self._answer) else (lambda _q: self._answer)
        out = []
        for q in questions:
            a = fn(q)
            if q.choices is None:
                out.append(Decision(q.id, p_yes=a if isinstance(a, (int, float)) else None))
            else:
                out.append(Decision(q.id, probs=dict(a) if isinstance(a, Mapping) else None))
        return out
