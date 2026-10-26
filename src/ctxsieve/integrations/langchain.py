"""`create_agent` middleware wrapping the LangGraph hooks.

    agent = create_agent(
        model, tools,
        middleware=[SlidingWindowMiddleware(decider, WindowPolicy(max_messages=20))],
    )

`before_model` runs the hook's `update()` (decides once per message, persists decisions in
state). In view mode `wrap_model_call` swaps the request messages for the rendered view; in
state mode the state itself already holds the pruned context.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.agents.middleware.types import OmitFromInput
from typing_extensions import NotRequired

from .langgraph import SlidingWindowHook, ToolPruneHook


class CtxSieveAgentState(AgentState):
    ctxsieve_window: Annotated[NotRequired[dict[str, Any]], OmitFromInput]
    ctxsieve_prune: Annotated[NotRequired[dict[str, Any]], OmitFromInput]


class _HookMiddleware(AgentMiddleware):
    state_schema = CtxSieveAgentState  # type: ignore[assignment]
    hook: SlidingWindowHook | ToolPruneHook

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.hook.update(state, getattr(runtime, "store", None))

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        # The hook makes blocking decider and store calls; run it off the event loop (sync
        # calls to async stores are allowed from a worker thread).
        return await asyncio.to_thread(self.hook.update, state, getattr(runtime, "store", None))

    def _viewed(self, request: ModelRequest) -> ModelRequest:
        if self.hook.mode != "view":
            return request
        return request.override(messages=self.hook.view(request.state, request.messages))

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        return handler(self._viewed(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        return await handler(self._viewed(request))


class SlidingWindowMiddleware(_HookMiddleware):
    """Sliding window with decided eviction (drop / pin / summarize). See SlidingWindowHook."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.hook = SlidingWindowHook(*args, **kwargs)


class ToolPruningMiddleware(_HookMiddleware):
    """Keep / truncate / clear older tool results. See ToolPruneHook."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.hook = ToolPruneHook(*args, **kwargs)
