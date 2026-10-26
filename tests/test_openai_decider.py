from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ctxsieve.deciders.base import Question
from ctxsieve.deciders.openai_decisions import MAX_QUESTIONS, OpenAIDecisionsDecider, to_decision_input


class FakeDecisions:
    """Mimics client.decisions; answers every predicate with a fixed probability."""

    def __init__(self, probability=0.7, refuse=()):
        self.probability = probability
        self.refuse = set(refuse)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        answers = [
            SimpleNamespace(name=q["name"], type="refusal")
            if q["name"] in self.refuse
            else SimpleNamespace(name=q["name"], type="predicate", probability=self.probability)
            for q in kwargs["questions"]
        ]
        usage = SimpleNamespace(input_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=40))
        return SimpleNamespace(answers=answers, model=kwargs["model"], usage=usage)


def make(**kw):
    fake = FakeDecisions(**kw)
    return OpenAIDecisionsDecider(client=SimpleNamespace(decisions=fake)), fake


def conversation():
    return [
        SystemMessage("sys"),
        HumanMessage("weather?"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "search", "args": {"q": "ist"}}]),
        ToolMessage("sunny", tool_call_id="c1", name="search"),
        AIMessage("It is sunny."),
    ]


def test_input_mapping():
    items = to_decision_input(conversation())
    assert items == [
        {"type": "message", "role": "system", "content": "[m0] sys"},
        {"type": "message", "role": "user", "content": "[m1] weather?"},
        {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"q": "ist"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "[m3] sunny"},
        {"type": "message", "role": "assistant", "content": "[m4] It is sunny."},
    ]


def test_predicates_and_refusal():
    decider, fake = make(probability=0.7, refuse={"q1"})
    out = decider.decide(conversation(), [Question("a", "A?"), Question("b", "B?")])
    assert [(d.question_id, d.p_yes) for d in out] == [("a", 0.7), ("b", None)]
    req = fake.requests[0]
    assert req["model"] == "gpt-6-luna"
    assert req["questions"][0] == {"type": "predicate", "name": "q0", "instructions": "A?"}
    assert decider.usage == {"requests": 1, "input_tokens": 100, "cached_tokens": 40}


def test_chunks_at_question_limit():
    decider, fake = make()
    qs = [Question(f"x{i}", "?") for i in range(MAX_QUESTIONS + 5)]
    out = decider.decide(conversation(), qs)
    assert [len(r["questions"]) for r in fake.requests] == [MAX_QUESTIONS, 5]
    assert [d.question_id for d in out] == [q.id for q in qs]


@pytest.mark.skipif(not os.environ.get("RUN_LIVE_OPENAI"), reason="set RUN_LIVE_OPENAI=1 and OPENAI_API_KEY")
def test_live_smoke():
    decider = OpenAIDecisionsDecider()
    q = Question("q", "The tool output in message [m3] says the weather is sunny.")
    [d] = decider.decide(conversation(), [q])
    assert d.p_yes is not None and 0.0 <= d.p_yes <= 1.0
