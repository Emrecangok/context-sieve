"""Decider backed by the OpenAI Decisions API (POST /v1/decisions, SDK >= 3.26)."""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .base import Decision, Question

MAX_QUESTIONS = 400  # per request, per the API docs


def tag(index: int) -> str:
    """Label prepended to message `index` so questions can refer to it."""
    return f"[m{index}]"


def to_decision_input(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        text = f"{tag(i)} {m.text}"
        if isinstance(m, SystemMessage):
            items.append({"type": "message", "role": "system", "content": text})
        elif isinstance(m, HumanMessage):
            items.append({"type": "message", "role": "user", "content": text})
        elif isinstance(m, AIMessage):
            if m.text or not m.tool_calls:
                items.append({"type": "message", "role": "assistant", "content": text})
            for j, tc in enumerate(m.tool_calls):
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id") or f"call_m{i}_{j}",
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False),
                })
        elif isinstance(m, ToolMessage):
            items.append({"type": "function_call_output", "call_id": m.tool_call_id, "output": text})
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
                questions=[
                    {"type": "predicate", "name": f"q{k}", "instructions": q.text}
                    for k, q in enumerate(chunk)
                ],
                **extra,
            )
            self._record_usage(resp)
            answers = {a.name: a for a in resp.answers}
            for k, q in enumerate(chunk):
                a = answers.get(f"q{k}")
                kind = getattr(a, "type", None)
                p = a.probability if kind == "predicate" else None
                out.append(Decision(q.id, p, raw={"type": kind, "model": resp.model}))
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
