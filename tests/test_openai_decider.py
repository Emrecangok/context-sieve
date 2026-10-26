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
        {"type": "message", "role": "assistant", "content": "[m2]"},
        {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"q": "ist"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "[m3] sunny"},
        {"type": "message", "role": "assistant", "content": "[m4] It is sunny."},
    ]


def test_every_tool_output_matches_an_earlier_call():
    # Review finding: the API rejects outputs without a matching call, which failed the whole batch.
    items = to_decision_input([
        ToolMessage("old", tool_call_id="c0"),  # call trimmed upstream
        HumanMessage("task"),
        AIMessage("", tool_calls=[
            {"id": "c1", "name": "search", "args": {}},
            {"id": None, "name": "calc", "args": {"x": 1}},
        ]),
        ToolMessage("r1", tool_call_id="c1"),
        ToolMessage("r2", tool_call_id="unknown"),
    ])
    calls = [it["call_id"] for it in items if it["type"] == "function_call"]
    outputs = [it["call_id"] for it in items if it["type"] == "function_call_output"]
    assert calls == ["c1"] and outputs == ["c1"]
    assert items[0] == {"type": "message", "role": "user", "content": "[m0] (result of tool call c0) old"}
    assert items[2] == {"type": "message", "role": "assistant", "content": '[m2] (tool call calc {"x": 1})'}
    assert items[-1]["content"] == "[m4] (result of tool call unknown) r2"
    for i in range(5):  # every message keeps its tag
        assert any(f"[m{i}]" in str(it.get("content", it.get("output", ""))) for it in items)


def test_predicates_and_refusal():
    decider, fake = make(probability=0.7, refuse={"q1"})
    out = decider.decide(conversation(), [Question("a", "A?"), Question("b", "B?")])
    assert [(d.question_id, d.p_yes) for d in out] == [("a", 0.7), ("b", None)]
    req = fake.requests[0]
    assert req["model"] == "gpt-6-luna"
    assert req["questions"][0] == {"type": "predicate", "name": "q0", "instructions": "A?"}
    assert decider.usage == {"requests": 1, "input_tokens": 100, "cached_tokens": 40}


def test_choice_questions():
    class ChoiceFake(FakeDecisions):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            q = kwargs["questions"][0]
            probs = [SimpleNamespace(value=c["value"], probability=p) for c, p in zip(q["choices"], (0.1, 0.7, 0.2))]
            answers = [SimpleNamespace(name=q["name"], type="choice", choice="pin", confidence=0.7, probabilities=probs)]
            return SimpleNamespace(answers=answers, model=kwargs["model"], usage=None)

    fake = ChoiceFake()
    decider = OpenAIDecisionsDecider(client=SimpleNamespace(decisions=fake))
    [d] = decider.decide(conversation(), [Question("u0", "What now?", choices=("drop", "pin", "summarize"))])
    assert d.probs == {"drop": 0.1, "pin": 0.7, "summarize": 0.2}
    assert d.p_yes is None
    assert fake.requests[0]["questions"][0] == {
        "type": "choice",
        "name": "q0",
        "instructions": "What now?",
        "choices": [{"value": "drop"}, {"value": "pin"}, {"value": "summarize"}],
    }


def test_choice_refusal_is_no_answer():
    decider, _ = make(refuse={"q0"})
    [d] = decider.decide(conversation(), [Question("u0", "?", choices=("a", "b"))])
    assert d.probs is None


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
