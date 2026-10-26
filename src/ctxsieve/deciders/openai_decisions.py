"""Decider backed by the OpenAI Decisions API (POST /v1/decisions, SDK >= 3.26)."""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .base import Decision, Question, tag

MAX_QUESTIONS = 400  # per request, per the API docs


def to_api_question(name: str, q: Question) -> dict[str, Any]:
    if q.choices is None:
        return {"type": "predicate", "name": name, "instructions": q.text}
    return {
        "type": "choice",
        "name": name,
        "instructions": q.text,
        "choices": [{"value": c} for c in q.choices],
    }


def from_api_answer(q: Question, a: Any, model: str) -> Decision:
    kind = getattr(a, "type", None)
    raw = {"type": kind, "model": model}
    if q.choices is None:
        return Decision(q.id, p_yes=a.probability if kind == "predicate" else None, raw=raw)
    if kind != "choice":
        return Decision(q.id, probs=None, raw=raw)
    return Decision(q.id, probs={p.value: p.probability for p in a.probabilities}, raw=raw)


def to_decision_input(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Map messages to Decisions input items.

    The API requires every tool output to match an earlier call by id, so tool calls without
    an id are written into the assistant text, and results whose call is not in the input
    (e.g. trimmed upstream) are sent as plain tagged messages.
    """
    items: list[dict[str, Any]] = []
    call_ids: set[str] = set()
    for i, m in enumerate(messages):
        text = f"{tag(i)} {m.text}" if m.text else tag(i)
        if isinstance(m, SystemMessage):
            items.append({"type": "message", "role": "system", "content": text})
        elif isinstance(m, HumanMessage):
            items.append({"type": "message", "role": "user", "content": text})
        elif isinstance(m, AIMessage):
            unnamed = [tc for tc in m.tool_calls if not tc.get("id")]
            for tc in unnamed:
                text += f" (tool call {tc['name']} {json.dumps(tc['args'], ensure_ascii=False)})"
            # Always emitted, even with no text, so the tag of a tool-call message is visible.
            items.append({"type": "message", "role": "assistant", "content": text})
            for tc in m.tool_calls:
                if tc.get("id"):
                    call_ids.add(tc["id"])
                    items.append({
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"], ensure_ascii=False),
                    })
        elif isinstance(m, ToolMessage):
            if m.tool_call_id in call_ids:
                items.append({"type": "function_call_output", "call_id": m.tool_call_id, "output": text})
            else:
                content = f"{tag(i)} (result of tool call {m.tool_call_id}) {m.text}"
                items.append({"type": "message", "role": "user", "content": content})
        else:
            items.append({"type": "message", "role": "user", "content": f"{tag(i)} ({m.type}) {m.text}"})
    return items


class OpenAIDecisionsDecider:
    def __init__(
        self,
        model: str = "gpt-6-luna",
        client: Any = None,
        prompt_cache_key: str | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self.model = model
        self.name = f"openai-decisions:{model}"
        self._prompt_cache_key = prompt_cache_key
        self.usage = {"requests": 0, "input_tokens": 0, "cached_tokens": 0}

    def decide(self, state: Sequence[BaseMessage], questions: Sequence[Question]) -> list[Decision]:
        input_items = to_decision_input(state)
        extra = {"prompt_cache_key": self._prompt_cache_key} if self._prompt_cache_key else {}
        out: list[Decision] = []
        for start in range(0, len(questions), MAX_QUESTIONS):
            chunk = questions[start:start + MAX_QUESTIONS]
            # Our ids may contain characters the API rejects in `name`; send positional names.
            resp = self._client.decisions.create(
                model=self.model,
                input=input_items,
                questions=[to_api_question(f"q{k}", q) for k, q in enumerate(chunk)],
                **extra,
            )
            self._record_usage(resp)
            answers = {a.name: a for a in resp.answers}
            out.extend(from_api_answer(q, answers.get(f"q{k}"), resp.model) for k, q in enumerate(chunk))
        return out

    def _record_usage(self, resp: Any) -> None:
        self.usage["requests"] += 1
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        self.usage["input_tokens"] += usage.input_tokens
        details = getattr(usage, "input_tokens_details", None)
        if details is not None:
            self.usage["cached_tokens"] += details.cached_tokens
