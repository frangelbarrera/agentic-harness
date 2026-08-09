"""Tests for the specialist JSON self-correction retry (opt-in).

When ``SpecialistConfig.max_json_retries > 0`` and the LLM returns a
tool-call-free response that is NOT valid JSON, the specialist feeds its
own output back with a "return ONLY JSON" prompt and re-calls the model
instead of failing immediately. Off by default so existing behavior is
unchanged.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator

from arnes.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMUsage
from arnes.specialists.base import get_default_specialist_registry
from arnes.thread.thread import Thread
from arnes.tools.base import ToolContext
from arnes.tools.registry import get_default_registry

GOOD_JSON = (
    '{"files": [{"path": "a.py", "language": "python", "content": "x = 1", '
    '"action": "create"}], "summary": "ok"}'
)


class RetryProvider(LLMProvider):
    """Returns queued responses verbatim; counts how many calls were made."""

    def __init__(self, responses: list[str]) -> None:
        self._queue: deque[str] = deque(responses)
        self.calls = 0
        # Tell the specialist it is already middleware-wrapped so it does not
        # double-wrap with its own VerificationLayer (which would change the
        # request/response shape). We only test the retry loop here.
        self._arnes_wrapped = True

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str = "mock",
        **kwargs: object,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=self._queue.popleft(),
            tool_calls=[],
            usage=LLMUsage(tokens_in=10, tokens_out=2, model=model),
            model=model,
        )

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str = "mock",
        **kwargs: object,
    ) -> AsyncIterator[LLMResponse]:
        yield await self.complete(messages, model=model)

    def list_models(self):
        return ["mock/anything"]


def _run(coder, provider, *, retries: int):
    import asyncio

    coder.config.max_json_retries = retries
    thread = Thread.create()
    ctx = ToolContext(thread_id=thread.id, specialist="@coder", metadata={"interactive": False})
    try:
        return asyncio.run(
            coder.run(
                {"spec": "Write add(a,b)."},
                ctx,
                provider=provider,
                tool_registry=get_default_registry(),
                model_override="mock/anything",
            )
        )
    finally:
        coder.config.max_json_retries = 0


def test_no_retries_default_fails_on_prose():
    """Default (0 retries): a prose response fails immediately, one call."""
    provider = RetryProvider(["-- not json prose --"])
    coder = get_default_specialist_registry().get("@coder")
    result = _run(coder, provider, retries=0)
    assert result["success"] is False
    assert "valid JSON" in str(result["error"])
    assert provider.calls == 1


def test_retries_self_correct_to_valid_json():
    """With 1 retry, a prose response is corrected into a valid one."""
    provider = RetryProvider(["-- prose first --", GOOD_JSON])
    coder = get_default_specialist_registry().get("@coder")
    result = _run(coder, provider, retries=1)
    assert result["success"] is True
    assert provider.calls == 2
    assert result["output"]["files"][0]["path"] == "a.py"


def test_retries_exhausted_still_reports_error():
    """With retries enabled but the model keeps failing, the final error is returned."""
    provider = RetryProvider(["-- one --", "-- two --", "-- three --"])
    coder = get_default_specialist_registry().get("@coder")
    result = _run(coder, provider, retries=2)
    assert result["success"] is False
    assert provider.calls == 3
