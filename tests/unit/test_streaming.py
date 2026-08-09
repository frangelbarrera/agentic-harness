"""Tests for real streaming in ARNES LLM providers and middleware.

Covers:
1. ``MockLLMProvider.stream_complete`` yields the full response in one chunk.
2. ``OllamaProvider.stream_complete`` reads NDJSON from a mocked httpx
   stream and yields token-by-token chunks + a final usage chunk.
3. ``LiteLLMProvider.stream_complete`` iterates a mocked
   ``litellm.acompletion(stream=True)`` result and yields token chunks +
   a final usage chunk.
4. ``TokenOptimizer.stream_complete`` is a thin passthrough (no cache
   population for streaming in v0.1).
5. ``CostGuard.stream_complete`` tracks cost on the final chunk and
   raises ``BudgetExceeded`` pre-flight when the budget is already
   exceeded.

The httpx and litellm dependencies are mocked via ``monkeypatch`` — no
network calls are made.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm.types.utils import Delta, ModelResponse, StreamingChoices, Usage

from arnes.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMUsage
from arnes.llm.litellm_provider import LiteLLMProvider
from arnes.llm.mock import MockLLMProvider
from arnes.llm.ollama import OllamaProvider
from arnes.middleware.cost_guard import BudgetExceeded, CostBudget, CostGuard
from arnes.middleware.token_optimizer import TokenOptimizer

# ============================================================
# Helpers — fake httpx streaming module
# ============================================================


def _build_fake_streaming_httpx(
    ndjson_lines: list[str],
) -> tuple[MagicMock, MagicMock]:
    """Build a fake ``httpx`` module that streams the given NDJSON lines.

    Mirrors the ``_build_fake_httpx`` helper in ``test_fix_ai.py`` but for
    the streaming API (``client.stream()`` + ``response.aiter_lines()``).

    Returns ``(fake_httpx, stream_mock)`` — assert on ``stream_mock.call_args``
    to inspect the request payload.
    """

    async def _aiter_lines() -> AsyncIterator[str]:
        for line in ndjson_lines:
            yield line

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock(return_value=None)
    # ``aiter_lines()`` returns an async generator — create it once; the
    # provider iterates it exactly once per stream_complete() call.
    fake_response.aiter_lines = MagicMock(return_value=_aiter_lines())

    # ``client.stream(method, url, json=payload)`` returns a context manager
    # whose ``__aenter__`` yields the response.
    fake_stream_cm = MagicMock()
    fake_stream_cm.__aenter__ = AsyncMock(return_value=fake_response)
    fake_stream_cm.__aexit__ = AsyncMock(return_value=None)

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=fake_stream_cm)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = MagicMock(return_value=fake_client)
    fake_httpx.ConnectError = type("ConnectError", (Exception,), {})
    fake_httpx.ConnectTimeout = type("ConnectTimeout", (Exception,), {})
    return fake_httpx, fake_client.stream


def _build_fake_streaming_httpx_connect_error() -> MagicMock:
    """Build a fake ``httpx`` module whose stream ``__aenter__`` raises ConnectError."""
    fake_httpx = MagicMock()
    connect_error = fake_httpx.ConnectError = type("ConnectError", (Exception,), {})
    fake_httpx.ConnectTimeout = type("ConnectTimeout", (Exception,), {})

    fake_stream_cm = MagicMock()
    fake_stream_cm.__aenter__ = AsyncMock(side_effect=connect_error("connection refused"))
    fake_stream_cm.__aexit__ = AsyncMock(return_value=None)

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=fake_stream_cm)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    fake_httpx.AsyncClient = MagicMock(return_value=fake_client)
    return fake_httpx


# ============================================================
# Helpers — fake litellm stream chunks
# ============================================================


def _make_stream_chunk(
    *,
    content: str = "",
    usage: Usage | None = None,
    model: str = "openai/gpt-4o",
) -> ModelResponse:
    """Construct a minimal litellm ``ModelResponse`` streaming chunk.

    Uses ``StreamingChoices`` + ``Delta`` so the attribute accesses in
    ``LiteLLMProvider.stream_complete()`` (``choices[0].delta.content``)
    match the real surface.
    """
    delta = Delta(content=content if content else None)
    return ModelResponse(
        id="test-chunk",
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=None)],
        model=model,
        usage=usage,
    )


def _patch_litellm_acompletion_stream(
    monkeypatch: pytest.MonkeyPatch,
    chunks: Sequence[Any],
) -> AsyncMock:
    """Replace ``litellm.acompletion`` with an AsyncMock returning a fake stream.

    The mock returns an async iterator that yields the given chunks. Returns
    the mock so the test can assert on ``call_args``.
    """
    import litellm

    async def _fake_stream() -> AsyncIterator[ModelResponse]:
        for chunk in chunks:
            yield chunk

    mock = AsyncMock(return_value=_fake_stream())
    monkeypatch.setattr(litellm, "acompletion", mock)
    return mock


# ============================================================
# Helpers — streaming provider with configurable cost
# ============================================================


class _StreamingCostProvider(LLMProvider):
    """Provider that streams token chunks and a final usage chunk.

    Used to test ``CostGuard.stream_complete`` cost tracking — the final
    chunk carries the full ``LLMUsage`` with ``cost_usd > 0``.
    """

    def __init__(
        self,
        *,
        tokens: list[str],
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        self.tokens = tokens
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = cost_usd

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            content="".join(self.tokens),
            tool_calls=[],
            usage=LLMUsage(
                tokens_in=self.tokens_in,
                tokens_out=self.tokens_out,
                cost_usd=self.cost_usd,
                model=model,
                cached=False,
            ),
            model=model,
        )

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        # Yield token chunks (zero usage — intermediate chunks carry no stats)
        for token in self.tokens:
            yield LLMResponse(
                content=token,
                tool_calls=[],
                usage=LLMUsage(
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    model=model,
                    cached=False,
                ),
                model=model,
            )
        # Yield final usage chunk (empty content, full usage)
        yield LLMResponse(
            content="",
            tool_calls=[],
            usage=LLMUsage(
                tokens_in=self.tokens_in,
                tokens_out=self.tokens_out,
                cost_usd=self.cost_usd,
                model=model,
                cached=False,
            ),
            model=model,
        )

    def list_models(self) -> list[str]:
        return ["mock"]


# ============================================================
# 1. MockLLMProvider.stream_complete
# ============================================================


class TestMockStreamComplete:
    @pytest.mark.asyncio
    async def test_mock_stream_yields_single_full_response_chunk(self):
        """``MockLLMProvider.stream_complete`` yields the full response in a
        single chunk — the default streaming implementation that lets
        callers write streaming-style code today and pick up real
        token-by-token streaming from other providers for free.
        """
        provider = MockLLMProvider(default_response="hello world")
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="hello, this is a test message")],
            model="mock",
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "hello world"
        assert chunks[0].model == "mock"
        # Usage is populated (mock estimates tokens from content length:
        # tokens_in = len(input) // 4, tokens_out = len(output) // 4)
        assert chunks[0].usage.tokens_in > 0
        assert chunks[0].usage.tokens_out > 0
        assert chunks[0].usage.cost_usd == 0.0  # Mock is free

    @pytest.mark.asyncio
    async def test_mock_stream_json_mode(self):
        """When ``response_format={"type": "json_object"}`` is set, the mock
        stream yields a JSON response — same behavior as ``complete()``.
        """
        provider = MockLLMProvider(default_response="ok")
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="give me json")],
            model="mock",
            response_format={"type": "json_object"},
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        # Content should be valid JSON
        data = json.loads(chunks[0].content)
        assert data["mock"] is True
        assert "input_hash" in data


# ============================================================
# 2. OllamaProvider.stream_complete
# ============================================================


class TestOllamaStreamComplete:
    @pytest.mark.asyncio
    async def test_ollama_stream_yields_token_by_token(self, monkeypatch):
        """``OllamaProvider.stream_complete`` reads NDJSON from
        ``/api/chat`` with ``stream: true`` and yields each token as a
        separate ``LLMResponse`` chunk. The final ``done: true`` line
        yields a chunk with the full ``LLMUsage``.
        """
        ndjson_lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": ", "}, "done": False}),
            json.dumps({"message": {"content": "world!"}, "done": False}),
            json.dumps(
                {
                    "message": {"content": ""},
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 3,
                }
            ),
        ]
        fake_httpx, stream_mock = _build_fake_streaming_httpx(ndjson_lines)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="ollama/llama3.2",
        ):
            chunks.append(chunk)

        # 3 token chunks + 1 final usage chunk
        assert len(chunks) == 4
        assert chunks[0].content == "Hello"
        assert chunks[1].content == ", "
        assert chunks[2].content == "world!"
        # Final chunk: empty content, full usage
        assert chunks[3].content == ""
        assert chunks[3].usage.tokens_in == 10
        assert chunks[3].usage.tokens_out == 3
        assert chunks[3].usage.cost_usd == 0.0  # Local = free
        assert chunks[3].model == "ollama/llama3.2"
        # Intermediate chunks have zero usage
        assert chunks[0].usage.tokens_in == 0
        assert chunks[0].usage.tokens_out == 0

        # Verify the request payload had stream=true
        assert stream_mock.called
        call_kwargs = stream_mock.call_args.kwargs
        payload = call_kwargs.get("json") or {}
        assert payload["stream"] is True
        assert payload["model"] == "llama3.2"  # vendor prefix stripped
        assert stream_mock.call_args.args[0] == "POST"

    @pytest.mark.asyncio
    async def test_ollama_stream_strips_vendor_prefix(self, monkeypatch):
        """When the model is given as ``ollama/llama3.2``, the stream
        payload must use just ``llama3.2`` (Ollama's API doesn't know
        about ARNES's vendor prefix). Mirrors ``complete()`` behavior.
        """
        ndjson_lines = [
            json.dumps({"message": {"content": "x"}, "done": False}),
            json.dumps(
                {"message": {"content": ""}, "done": True, "eval_count": 1, "prompt_eval_count": 1}
            ),
        ]
        fake_httpx, stream_mock = _build_fake_streaming_httpx(ndjson_lines)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        async for _ in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="ollama/llama3.2",
        ):
            pass

        payload = stream_mock.call_args.kwargs.get("json") or {}
        assert payload["model"] == "llama3.2"

    @pytest.mark.asyncio
    async def test_ollama_stream_passes_tools_in_payload(self, monkeypatch):
        """The ``tools`` kwarg must reach the Ollama stream payload —
        dropping it would silently disable tool-calling for streaming
        ReAct loops.
        """
        ndjson_lines = [
            json.dumps(
                {"message": {"content": ""}, "done": True, "eval_count": 0, "prompt_eval_count": 0}
            ),
        ]
        fake_httpx, stream_mock = _build_fake_streaming_httpx(ndjson_lines)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        tools = [{"type": "function", "function": {"name": "echo", "parameters": {}}}]
        async for _ in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="ollama/llama3.2",
            tools=tools,
        ):
            pass

        payload = stream_mock.call_args.kwargs.get("json") or {}
        assert payload["tools"] == tools

    @pytest.mark.asyncio
    async def test_ollama_stream_handles_trailing_content_on_done_chunk(self, monkeypatch):
        """When the ``done: true`` chunk also carries a trailing content
        token, both the token chunk and the usage chunk must be yielded
        — no tokens should be lost.
        """
        ndjson_lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps(
                {
                    "message": {"content": "!"},
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                }
            ),
        ]
        fake_httpx, _ = _build_fake_streaming_httpx(ndjson_lines)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="ollama/llama3.2",
        ):
            chunks.append(chunk)

        # "Hello" token + "!" trailing token + final usage chunk
        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == "!"
        assert chunks[2].content == ""
        assert chunks[2].usage.tokens_in == 5
        assert chunks[2].usage.tokens_out == 2

    @pytest.mark.asyncio
    async def test_ollama_stream_skips_malformed_ndjson_lines(self, monkeypatch):
        """Malformed NDJSON lines must be skipped, not abort the stream —
        a single bad line should not discard tokens already received.
        """
        ndjson_lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            "this is not valid json",
            json.dumps({"message": {"content": "world"}, "done": False}),
            json.dumps(
                {"message": {"content": ""}, "done": True, "eval_count": 2, "prompt_eval_count": 1}
            ),
        ]
        fake_httpx, _ = _build_fake_streaming_httpx(ndjson_lines)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="ollama/llama3.2",
        ):
            chunks.append(chunk)

        # "Hello" + "world" + final usage = 3 chunks (malformed line skipped)
        contents = [c.content for c in chunks if c.content]
        assert "Hello" in contents
        assert "world" in contents
        # Final chunk has usage
        assert chunks[-1].usage.tokens_out == 2

    @pytest.mark.asyncio
    async def test_ollama_stream_wraps_connect_error(self, monkeypatch):
        """When ``httpx.ConnectError`` is raised (Ollama not running), the
        provider must wrap it in a ``RuntimeError`` with install
        instructions — same behavior as ``complete()``.
        """
        fake_httpx = _build_fake_streaming_httpx_connect_error()
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            async for _ in provider.stream_complete(
                [LLMMessage(role="user", content="hi")],
                model="ollama/llama3.2",
            ):
                pass

    @pytest.mark.asyncio
    async def test_ollama_stream_yields_sentinel_when_no_done_chunk(self, monkeypatch):
        """If the stream ends without a ``done: true`` line (network
        interruption), a sentinel final chunk with zeroed usage is
        yielded so callers don't hang waiting for usage stats.
        """
        ndjson_lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            # No done=true line — stream just ends
        ]
        fake_httpx, _ = _build_fake_streaming_httpx(ndjson_lines)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="ollama/llama3.2",
        ):
            chunks.append(chunk)

        # 1 token chunk + 1 sentinel final chunk
        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == ""
        assert chunks[1].usage.tokens_in == 0
        assert chunks[1].usage.tokens_out == 0


# ============================================================
# 3. LiteLLMProvider.stream_complete
# ============================================================


class TestLiteLLMStreamComplete:
    @pytest.mark.asyncio
    async def test_litellm_stream_yields_token_by_token(self, monkeypatch):
        """``LiteLLMProvider.stream_complete`` iterates the
        ``CustomStreamWrapper`` from ``litellm.acompletion(stream=True)``
        and yields each ``delta.content`` as a separate chunk. A final
        chunk with the full ``LLMUsage`` is yielded after the stream ends.
        """
        chunks_data = [
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(content=", "),
            _make_stream_chunk(content="world!"),
            _make_stream_chunk(usage=Usage(prompt_tokens=10, completion_tokens=3)),
        ]
        mock = _patch_litellm_acompletion_stream(monkeypatch, chunks_data)

        provider = LiteLLMProvider()
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="openai/gpt-4o",
        ):
            chunks.append(chunk)

        # 3 content chunks + 1 final usage chunk
        assert len(chunks) == 4
        assert chunks[0].content == "Hello"
        assert chunks[1].content == ", "
        assert chunks[2].content == "world!"
        # Final chunk: empty content, full usage
        assert chunks[3].content == ""
        assert chunks[3].usage.tokens_in == 10
        assert chunks[3].usage.tokens_out == 3
        assert chunks[3].model == "openai/gpt-4o"
        # Cost is calculated from the pricing table
        assert chunks[3].usage.cost_usd > 0

        # Verify acompletion was called with stream=True
        assert mock.called
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["model"] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_litellm_stream_forwards_tools_and_response_format(self, monkeypatch):
        """``tools`` and ``response_format`` must reach litellm — same
        forwarding contract as ``complete()``.
        """
        chunks_data = [
            _make_stream_chunk(content="ok"),
            _make_stream_chunk(usage=Usage(prompt_tokens=5, completion_tokens=1)),
        ]
        mock = _patch_litellm_acompletion_stream(monkeypatch, chunks_data)

        provider = LiteLLMProvider()
        tools = [{"type": "function", "function": {"name": "echo"}}]
        async for _ in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="openai/gpt-4o",
            tools=tools,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=100,
        ):
            pass

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["tools"] == tools
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_litellm_stream_without_usage_yields_no_final_chunk(self, monkeypatch):
        """When the vendor doesn't stream usage (e.g. OpenAI without
        ``stream_options={"include_usage": True}``), no final usage chunk
        is yielded — callers detect this by the absence of a chunk with
        non-zero ``usage.tokens_out``.
        """
        chunks_data = [
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(content=" world"),
            # No usage on any chunk
        ]
        _patch_litellm_acompletion_stream(monkeypatch, chunks_data)

        provider = LiteLLMProvider()
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="openai/gpt-4o",
        ):
            chunks.append(chunk)

        # 2 content chunks, no final usage chunk
        assert len(chunks) == 2
        assert all(c.usage.tokens_in == 0 for c in chunks)
        assert all(c.usage.tokens_out == 0 for c in chunks)

    @pytest.mark.asyncio
    async def test_litellm_stream_assembles_tool_calls_from_fragments(self, monkeypatch):
        """Streamed ``delta.tool_calls`` fragments must be reassembled into a
        single tool_calls chunk, or the ReAct loop in streaming mode would
        silently never see any tool call from a hosted provider.
        """
        from types import SimpleNamespace

        def frag(tool_calls: list[dict[str, object]]) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(delta={"content": None, "tool_calls": tool_calls})]
            )

        chunks_data: list[object] = [
            frag(
                [
                    {
                        "index": 0,
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ]
            ),
            frag([{"index": 0, "function": {"arguments": '{"city": "Madri'}}]),
            frag([{"index": 0, "function": {"arguments": 'd"}'}}]),
        ]
        _patch_litellm_acompletion_stream(monkeypatch, chunks_data)

        provider = LiteLLMProvider()
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="weather in Madrid?")],
            model="openai/gpt-4o",
        ):
            chunks.append(chunk)

        with_tools = [c for c in chunks if c.tool_calls]
        assert len(with_tools) == 1
        call = with_tools[0].tool_calls[0]
        assert call["id"] == "call_123"
        assert call["function"]["name"] == "get_weather"
        # Fragments reassembled in order into one JSON-arguments string.
        assert call["function"]["arguments"] == '{"city": "Madrid"}'

    @pytest.mark.asyncio
    async def test_litellm_stream_handles_plain_delta_without_tool_calls(self, monkeypatch):
        """Chunks with neither content nor tool_calls are simply skipped."""
        from types import SimpleNamespace

        chunk = SimpleNamespace(choices=[SimpleNamespace(delta={"content": None})])
        _patch_litellm_acompletion_stream(monkeypatch, [chunk])
        provider = LiteLLMProvider()
        chunks = [c async for c in provider.stream_complete(
            [LLMMessage(role="user", content="hi")], model="openai/gpt-4o")]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_litellm_stream_merges_init_kwargs(self, monkeypatch):
        """Construction-time kwargs (e.g. ``api_key``) must reach litellm
        in streaming mode — same precedence as ``complete()``.
        """
        chunks_data = [_make_stream_chunk(content="ok")]
        mock = _patch_litellm_acompletion_stream(monkeypatch, chunks_data)

        provider = LiteLLMProvider(api_key="sk-init", base_url="https://init.example")
        async for _ in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="openai/gpt-4o",
            api_key="sk-override",
        ):
            pass

        call_kwargs = mock.call_args.kwargs
        # Per-call kwarg wins over init kwarg
        assert call_kwargs["api_key"] == "sk-override"
        # Init kwarg is preserved when not overridden
        assert call_kwargs["base_url"] == "https://init.example"

    @pytest.mark.asyncio
    async def test_litellm_stream_handles_empty_delta_content(self, monkeypatch):
        """Chunks with ``delta.content=None`` or empty string must be
        skipped (not yielded as empty-content chunks) — vendors send
        these for role markers and other metadata.
        """
        chunks_data = [
            _make_stream_chunk(content=""),  # empty — skip
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(content=""),  # empty — skip
            _make_stream_chunk(content=" world"),
        ]
        _patch_litellm_acompletion_stream(monkeypatch, chunks_data)

        provider = LiteLLMProvider()
        chunks: list[LLMResponse] = []
        async for chunk in provider.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="openai/gpt-4o",
        ):
            chunks.append(chunk)

        # Only non-empty content chunks are yielded
        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"


# ============================================================
# 4. TokenOptimizer.stream_complete passthrough
# ============================================================


class TestTokenOptimizerStreamPassthrough:
    @pytest.mark.asyncio
    async def test_passthrough_yields_inner_provider_chunks(self):
        """``TokenOptimizer.stream_complete`` must pass through all chunks
        from the inner provider unchanged — no transformation, no
        filtering.
        """
        inner = _StreamingCostProvider(
            tokens=["Hello", " world"],
            tokens_in=10,
            tokens_out=2,
            cost_usd=0.0,
        )
        optimizer = TokenOptimizer(inner, enable_cache=True, enable_routing=False)

        chunks: list[LLMResponse] = []
        async for chunk in optimizer.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="mock/test",
        ):
            chunks.append(chunk)

        # 2 token chunks + 1 final usage chunk — all passed through
        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].usage.tokens_in == 10
        assert chunks[2].usage.tokens_out == 2

    @pytest.mark.asyncio
    async def test_passthrough_does_not_populate_cache(self):
        """Streaming calls must NOT populate the semantic cache in v0.1 —
        caching a stream requires reassembling the full response first,
        which defeats the latency benefit of streaming (v0.2 work).
        """
        inner = MockLLMProvider(default_response="test response")
        optimizer = TokenOptimizer(inner, enable_cache=True, enable_routing=False)

        async for _ in optimizer.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="mock/test",
        ):
            pass

        stats = optimizer.stats()
        assert stats["cache_size"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

    @pytest.mark.asyncio
    async def test_passthrough_does_not_increment_counters(self):
        """Streaming calls must NOT touch ``cache_hits`` / ``cache_misses``
        — the cache is bypassed entirely for streaming in v0.1.
        """
        inner = MockLLMProvider(default_response="test")
        optimizer = TokenOptimizer(inner, enable_cache=True, enable_routing=False)

        # Make a streaming call
        async for _ in optimizer.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="mock/test",
        ):
            pass

        # Now make a non-streaming call — it should be a cache miss
        # (streaming didn't populate the cache)
        await optimizer.complete(
            [LLMMessage(role="user", content="hi")],
            model="mock/test",
        )

        stats = optimizer.stats()
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 0


# ============================================================
# 5. CostGuard.stream_complete tracks cost
# ============================================================


class TestCostGuardStreamTracking:
    @pytest.mark.asyncio
    async def test_tracks_cost_on_final_chunk(self):
        """``CostGuard.stream_complete`` must update ``spent_usd`` and
        ``calls_made`` after the stream ends, using the final chunk's
        ``usage.cost_usd``.
        """
        provider = _StreamingCostProvider(
            tokens=["Hello", " world"],
            tokens_in=10,
            tokens_out=2,
            cost_usd=0.001,
        )
        guard = CostGuard(provider, budget=CostBudget(task_budget_usd=1.0))

        chunks: list[LLMResponse] = []
        async for chunk in guard.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="mock/test",
        ):
            chunks.append(chunk)

        # All chunks pass through (2 tokens + 1 final usage)
        assert len(chunks) == 3
        assert guard.calls_made == 1
        assert guard.spent_usd == pytest.approx(0.001)

    @pytest.mark.asyncio
    async def test_stream_with_zero_cost_still_counts_call(self):
        """When the final chunk has ``cost_usd=0`` (e.g. local Ollama),
        ``calls_made`` must still increment but ``spent_usd`` must not
        change.
        """
        provider = _StreamingCostProvider(
            tokens=["hi"],
            tokens_in=5,
            tokens_out=1,
            cost_usd=0.0,
        )
        guard = CostGuard(provider, budget=CostBudget(task_budget_usd=1.0))

        async for _ in guard.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="mock/test",
        ):
            pass

        assert guard.calls_made == 1
        assert guard.spent_usd == 0.0

    @pytest.mark.asyncio
    async def test_stream_aborts_when_budget_already_exceeded(self):
        """If ``spent_usd >= abort_threshold`` before the stream starts,
        ``BudgetExceeded`` must be raised on the first iteration — no
        tokens are streamed, no provider call is made.
        """
        provider = _StreamingCostProvider(
            tokens=["hi"],
            tokens_in=1,
            tokens_out=1,
            cost_usd=0.0,
        )
        guard = CostGuard(provider, budget=CostBudget(task_budget_usd=0.10))
        guard.spent_usd = 0.11  # over the $0.10 budget

        with pytest.raises(BudgetExceeded, match="Budget exceeded"):
            async for _ in guard.stream_complete(
                [LLMMessage(role="user", content="hi")],
                model="mock/test",
            ):
                pass

        # Provider must NOT have been called
        assert guard.calls_made == 0
        assert guard._aborted is True

    @pytest.mark.asyncio
    async def test_stream_raises_when_already_aborted(self):
        """Once ``_aborted`` is set (by a prior budget breach), subsequent
        streaming calls must raise immediately without invoking the
        provider.
        """
        provider = _StreamingCostProvider(
            tokens=["hi"],
            tokens_in=1,
            tokens_out=1,
            cost_usd=0.0,
        )
        guard = CostGuard(provider, budget=CostBudget(task_budget_usd=1.0))
        guard._aborted = True

        with pytest.raises(BudgetExceeded, match="aborted"):
            async for _ in guard.stream_complete(
                [LLMMessage(role="user", content="hi")],
                model="mock/test",
            ):
                pass

        assert guard.calls_made == 0

    @pytest.mark.asyncio
    async def test_stream_circuit_breaker_trips(self):
        """The circuit breaker must trip on streaming calls too — if the
        spend rate exceeds ``max_usd_per_minute``, ``BudgetExceeded`` is
        raised before the stream starts.
        """
        import time

        provider = _StreamingCostProvider(
            tokens=["hi"],
            tokens_in=1,
            tokens_out=1,
            cost_usd=0.0,
        )
        guard = CostGuard(
            provider,
            budget=CostBudget(task_budget_usd=10.0, max_usd_per_minute=0.001),
        )
        # Inject fake spend to trip the circuit breaker
        guard._spend_history.append((time.time(), 0.005))

        with pytest.raises(BudgetExceeded, match="Circuit breaker"):
            async for _ in guard.stream_complete(
                [LLMMessage(role="user", content="hi")],
                model="mock/test",
            ):
                pass

        assert guard._aborted is True
        assert guard.calls_made == 0

    @pytest.mark.asyncio
    async def test_stream_accumulates_multiple_calls(self):
        """Multiple streaming calls must accumulate ``spent_usd`` and
        ``calls_made`` — each call's final-chunk cost is added to the
        running total.
        """
        provider = _StreamingCostProvider(
            tokens=["a"],
            tokens_in=5,
            tokens_out=1,
            cost_usd=0.002,
        )
        guard = CostGuard(provider, budget=CostBudget(task_budget_usd=1.0))

        for _ in range(3):
            async for _ in guard.stream_complete(
                [LLMMessage(role="user", content="hi")],
                model="mock/test",
            ):
                pass

        assert guard.calls_made == 3
        assert guard.spent_usd == pytest.approx(0.006)

    @pytest.mark.asyncio
    async def test_stream_through_middleware_stack(self):
        """End-to-end: stream through the full middleware stack
        (CostGuard → TokenOptimizer → inner provider) and verify cost
        tracking still works.
        """
        inner = _StreamingCostProvider(
            tokens=["Hello", " world"],
            tokens_in=10,
            tokens_out=2,
            cost_usd=0.001,
        )
        optimizer = TokenOptimizer(inner, enable_cache=True, enable_routing=False)
        guard = CostGuard(optimizer, budget=CostBudget(task_budget_usd=1.0))

        chunks: list[LLMResponse] = []
        async for chunk in guard.stream_complete(
            [LLMMessage(role="user", content="hi")],
            model="mock/test",
        ):
            chunks.append(chunk)

        # All chunks passed through the full stack
        assert len(chunks) == 3
        assert guard.calls_made == 1
        assert guard.spent_usd == pytest.approx(0.001)
        # TokenOptimizer cache was NOT populated (streaming bypasses cache)
        assert optimizer.stats()["cache_size"] == 0
