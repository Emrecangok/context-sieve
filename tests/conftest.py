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
