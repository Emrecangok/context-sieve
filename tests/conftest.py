from __future__ import annotations

from hypothesis import strategies as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

text = st.text(alphabet="abcdefghij klmnop", min_size=0, max_size=800)


@st.composite
def conversations(draw) -> list[BaseMessage]:
    """System? + first human + a mix of chat turns and tool-call blocks."""
    msgs: list[BaseMessage] = []
    if draw(st.booleans()):
        msgs.append(SystemMessage("You are a helpful agent."))
    msgs.append(HumanMessage(draw(text) or "task"))
    for _ in range(draw(st.integers(0, 8))):
        kind = draw(st.sampled_from(["human", "ai", "tools"]))
        if kind == "human":
            msgs.append(HumanMessage(draw(text)))
        elif kind == "ai":
            msgs.append(AIMessage(draw(text)))
        else:
            n = draw(st.integers(1, 3))
            base = len(msgs)
            calls = [{"id": f"c{base}_{k}", "name": "search", "args": {"q": str(k)}} for k in range(n)]
            msgs.append(AIMessage("", tool_calls=calls))
            for c in calls:
                msgs.append(ToolMessage(draw(text), tool_call_id=c["id"], name="search"))
    for i, m in enumerate(msgs):
        m.id = f"m{i}"
    return msgs


class FakeSummarizer:
    name = "fake-summarizer"

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def summarize(self, messages, previous):
        self.calls.append(("summarize", [m.id for m in messages], list(previous)))
        if self.error:
            raise self.error
        return "S(" + ",".join(str(m.id) for m in messages) + ")"

    def merge(self, sections):
        self.calls.append(("merge", list(sections)))
        if self.error:
            raise self.error
        return "M(" + "+".join(sections) + ")"


class ContentDecider:
    """Answers from the text of the tagged messages, so tests can steer decisions.

    Choice (eviction): PIN -> pin, DROP -> drop, otherwise `default_choice` (summarize).
    Predicate (tool pruning): NOISE -> 0.0 (not needed), otherwise 0.9.
    """

    name = "content"

    def __init__(self, default_choice="summarize"):
        self.default_choice = default_choice
        self.calls = []

    def decide(self, state, questions):
        import re

        from ctxsieve.deciders.base import Decision

        self.calls.append(list(questions))
        out = []
        for q in questions:
            tags = [int(t) for t in re.findall(r"\[m(\d+)\]", q.text)]
            text = " ".join(state[i].text for i in range(min(tags), max(tags) + 1))
            if q.choices:
                label = "pin" if "PIN" in text else "drop" if "DROP" in text else self.default_choice
                out.append(Decision(q.id, probs={c: 0.9 if c == label else 0.05 for c in q.choices}))
            else:
                out.append(Decision(q.id, p_yes=0.0 if "NOISE" in text else 0.9))
        return out


def chunks(msgs, cuts):
    """Split `msgs` into consecutive chunks at the given (wrapped) positions."""
    bounds = sorted({c % (len(msgs) + 1) for c in cuts} | {0, len(msgs)})
    return [msgs[a:b] for a, b in zip(bounds, bounds[1:])]
