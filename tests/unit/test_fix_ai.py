"""Tests for the ARNES AI layer — critical specialist and provider behaviours.

Covers:
1. OllamaProvider passes `tools` and parses `tool_calls` from the response.
2. VerificationLayer skips hedging detection when JSON mode is active.
3. Specialist.run() returns a clear error when max_iterations is exceeded.
4. Bonus: _clean_json_response strips ```json fences.
5. Bonus: LiteLLMProvider.peek_cost returns a non-None estimate.
6. Bonus: @reviewer now carries a pydantic_model and validates through it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from arnes.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMUsage
from arnes.llm.ollama import OllamaProvider
from arnes.middleware.verification import VerificationConfig, VerificationLayer
from arnes.specialists.base import Specialist, SpecialistConfig
from arnes.specialists.reviewer import Reviewer, ReviewerOutput
from arnes.tools.base import ToolContext

# ============================================================
# 1. OllamaProvider — tools + tool_calls
# ============================================================


def _build_fake_httpx(response_payload: dict[str, Any]) -> tuple[MagicMock, MagicMock]:
    """Build a fake httpx module that captures the request payload and
    returns the given response payload.

    Returns (fake_httpx_module, post_mock) — assert on ``post_mock.call_args``
    to inspect the JSON body the provider sent to /api/chat.
    """
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock(return_value=None)
    fake_resp.json = MagicMock(return_value=response_payload)

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = MagicMock(return_value=fake_client)
    fake_httpx.ConnectError = type("ConnectError", (Exception,), {})  # for the except clause
    fake_httpx.ConnectTimeout = type("ConnectTimeout", (Exception,), {})
    return fake_httpx, fake_client.post


class TestOllamaProvider:
    @pytest.mark.asyncio
    async def test_tools_passed_to_ollama_payload(self, monkeypatch):
        """The `tools` kwarg must reach the Ollama API payload (not be dropped)."""
        fake_httpx, post_mock = _build_fake_httpx(
            {"message": {"content": "hi"}, "eval_count": 1, "prompt_eval_count": 1}
        )
        monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        tools = [{"type": "function", "function": {"name": "echo", "parameters": {}}}]
        await provider.complete(
            [LLMMessage(role="user", content="hi")],
            model="ollama/llama3.2",
            tools=tools,
        )

        assert post_mock.called, "OllamaProvider did not POST anything"
        payload = post_mock.call_args.kwargs.get("json") or {}
        assert payload["tools"] == tools, (
            "OllamaProvider dropped the `tools` parameter — tools must be passed "
            "through to the Ollama API payload (supported since v0.3.0)."
        )

    @pytest.mark.asyncio
    async def test_tool_calls_parsed_from_response(self, monkeypatch):
        """tool_calls in the Ollama response must be parsed into LLMResponse.tool_calls."""
        ollama_response = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "fs_read",
                            "arguments": {"path": "src/foo.py"},
                        }
                    }
                ],
            },
            "eval_count": 5,
            "prompt_eval_count": 10,
        }
        fake_httpx, _ = _build_fake_httpx(ollama_response)
        monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        response = await provider.complete(
            [LLMMessage(role="user", content="read foo")],
            model="ollama/llama3.2",
            tools=[{"type": "function", "function": {"name": "fs_read", "parameters": {}}}],
        )

        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc["function"]["name"] == "fs_read"
        # OpenAI shape: arguments is a JSON string
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"path": "src/foo.py"}
        assert tc["type"] == "function"
        assert "id" in tc

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_empty_list(self, monkeypatch):
        """If Ollama returns no tool_calls field, the response tool_calls must be []."""
        ollama_response = {
            "message": {"content": "just text, no tools"},
            "eval_count": 1,
            "prompt_eval_count": 1,
        }
        fake_httpx, _ = _build_fake_httpx(ollama_response)
        monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

        provider = OllamaProvider()
        response = await provider.complete(
            [LLMMessage(role="user", content="hi")],
            model="ollama/llama3.2",
            tools=[{"type": "function", "function": {"name": "fs_read", "parameters": {}}}],
        )

        assert response.tool_calls == []
        assert response.content == "just text, no tools"


# ============================================================
# 2. VerificationLayer — hedging skipped in JSON mode
# ============================================================


class _StaticProvider(LLMProvider):
    """Trivial provider that always returns the same string content."""

    def __init__(self, content: str) -> None:
        self._content = content

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
            content=self._content,
            tool_calls=[],
            usage=LLMUsage(tokens_in=10, tokens_out=5, cost_usd=0.0, model=model, cached=False),
            model=model,
        )

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Yield the full response in one chunk (matches MockLLMProvider contract)."""
        response = await self.complete(
            messages,
            model=model,
            tools=tools,
            response_format=response_format,
            response_schema=response_schema,
            **kwargs,
        )
        yield response

    def list_models(self) -> list[str]:
        return ["mock/test"]


class TestHedgingJSONMode:
    @pytest.mark.asyncio
    async def test_hedging_skipped_when_json_mode_active(self):
        """A valid JSON response with hedging phrases inside a string field
        must NOT trigger hedging detection when JSON mode is active."""
        # The content contains "I'm not sure" — would normally trigger hedging.
        # But because we're in JSON mode + have a response_schema, the schema
        # check is the guard, not the hedging regex.
        provider = _StaticProvider(
            '{"summary": "I\'m not sure about the auth flow", "result": "ok"}'
        )
        verification = VerificationLayer(
            provider,
            VerificationConfig(structured_outputs=True, refusal_pattern=True, detect_hedging=True),
        )

        response = await verification.complete(
            [LLMMessage(role="user", content="What is the auth flow?")],
            model="mock/test",
            response_schema={
                "type": "object",
                "required": ["summary", "result"],
                "properties": {"summary": {"type": "string"}, "result": {"type": "string"}},
            },
        )

        # Should NOT be replaced with the refusal message — hedging was skipped
        # because JSON mode was active and the JSON validated successfully.
        assert "don't have enough confidence" not in response.content
        assert "I'm not sure about the auth flow" in response.content

    @pytest.mark.asyncio
    async def test_hedging_still_runs_without_json_mode(self):
        """Hedging detection still triggers for free-text responses (no schema)."""
        provider = _StaticProvider("I'm not sure about the answer")
        verification = VerificationLayer(
            provider,
            VerificationConfig(structured_outputs=False, refusal_pattern=True, detect_hedging=True),
        )

        response = await verification.complete(
            [LLMMessage(role="user", content="What is X?")],
            model="mock/test",
        )

        assert (
            response.content
            == "I don't have enough confidence to answer this. Please verify manually."
        )


# ============================================================
# 3. Specialist — max_iterations exceeded
# ============================================================


class _AlwaysToolCallProvider(LLMProvider):
    """Provider that ALWAYS returns a tool_call, never a final response."""

    def __init__(self) -> None:
        self.call_count = 0

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
        self.call_count += 1
        return LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": f"call_{self.call_count}",
                    "type": "function",
                    "function": {
                        # Call a non-existent tool so no real tool runs; the
                        # specialist will record the failure and loop again.
                        "name": "noop_tool",
                        "arguments": "{}",
                    },
                }
            ],
            usage=LLMUsage(tokens_in=10, tokens_out=5, cost_usd=0.0, model=model, cached=False),
            model=model,
        )

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Yield the full response in one chunk (matches MockLLMProvider contract)."""
        response = await self.complete(
            messages,
            model=model,
            tools=tools,
            response_format=response_format,
            response_schema=response_schema,
            **kwargs,
        )
        yield response

    def list_models(self) -> list[str]:
        return ["mock/test"]


class _MinimalSpecialist(Specialist):
    """Specialist with no tools and no schema — used to test the loop bound."""

    config: ClassVar[SpecialistConfig] = SpecialistConfig(
        name="@test-minimal",
        description="minimal specialist for tests",
        system_prompt="You are a test specialist.",
        tools=[],
        output_schema=None,
        pydantic_model=None,
        max_iterations=3,
        default_model="mock/test",
    )


class TestSpecialistMaxIterations:
    @pytest.mark.asyncio
    async def test_max_iterations_exceeded_returns_clear_error(self):
        """When the LLM keeps calling tools past max_iterations, the specialist
        must return a clear error rather than validating an empty response."""
        from uuid import uuid4

        specialist = _MinimalSpecialist()
        ctx = ToolContext(thread_id=uuid4(), metadata={"interactive": False})
        provider = _AlwaysToolCallProvider()

        result = await specialist.run({"task": "x"}, ctx, provider=provider)

        assert result["success"] is False
        assert "max_iterations" in result["error"]
        assert "3" in result["error"]
        # The provider should have been called exactly max_iterations times.
        assert provider.call_count == 3
        # Must NOT have a phantom 'output' field — that would mean we validated
        # an empty/intermediate tool-call response as if it were the final answer.
        assert "output" not in result
        assert "tool_results" in result


# ============================================================
# 4. Bonus: _clean_json_response helper
# ============================================================


class TestCleanJsonResponse:
    def test_strips_json_fenced_block(self):
        text = '```json\n{"verdict": "approve"}\n```'
        assert Specialist._clean_json_response(text) == '{"verdict": "approve"}'

    def test_strips_bare_fenced_block(self):
        text = '```\n{"a": 1}\n```'
        assert Specialist._clean_json_response(text) == '{"a": 1}'

    def test_no_fences_untouched(self):
        text = '{"a": 1}'
        assert Specialist._clean_json_response(text) == '{"a": 1}'

    def test_strips_language_tag(self):
        text = '```jsonl\n{"x": 1}\n```'
        assert Specialist._clean_json_response(text) == '{"x": 1}'

    def test_single_line_fence(self):
        # Llama 3.2 sometimes does ```{"x": 1}```
        text = '```{"x": 1}```'
        assert Specialist._clean_json_response(text) == '{"x": 1}'

    def test_non_string_returned_unchanged(self):
        assert Specialist._clean_json_response(None) is None  # type: ignore[arg-type]


# ============================================================
# 4b. Bonus: _parse_and_validate_output edge cases
# (covers the JSON-parse-error, missing-required-fields, and empty-response
# branches in specialists/base.py)
# ============================================================


class _SchemaSpecialist(Specialist):
    """Specialist that declares an output_schema — used to exercise the
    JSON-parse-error and missing-required-fields code paths."""

    config: ClassVar[SpecialistConfig] = SpecialistConfig(
        name="@test-schema",
        description="schema specialist for tests",
        system_prompt="You are a test specialist.",
        tools=[],
        output_schema={
            "type": "object",
            "required": ["result"],
            "properties": {"result": {"type": "string"}},
        },
        max_iterations=1,
        default_model="mock/test",
    )


class _PydanticSpecialist(Specialist):
    """Specialist that declares a pydantic_model — covers the
    pydantic-validation-success path."""

    class _Output(BaseModel):
        result: str

    config: ClassVar[SpecialistConfig] = SpecialistConfig(
        name="@test-pydantic",
        description="pydantic specialist for tests",
        system_prompt="You are a test specialist.",
        tools=[],
        pydantic_model=_Output,
        max_iterations=1,
        default_model="mock/test",
    )


class TestParseAndValidateOutput:
    def _make_response(self, content: str) -> LLMResponse:
        return LLMResponse(
            content=content,
            tool_calls=[],
            usage=LLMUsage(tokens_in=10, tokens_out=5, cost_usd=0.0, model="mock/test"),
            model="mock/test",
        )

    def test_empty_response_returns_error(self):
        specialist = _SchemaSpecialist()
        result = specialist._parse_and_validate_output(
            self._make_response(""),
            LLMUsage(),
            [],
        )
        assert result["success"] is False
        assert "Empty response" in result["error"]

    def test_malformed_json_with_schema_returns_error(self):
        specialist = _SchemaSpecialist()
        result = specialist._parse_and_validate_output(
            self._make_response("not valid json at all"),
            LLMUsage(),
            [],
        )
        assert result["success"] is False
        assert "did not return valid JSON" in result["error"]

    def test_missing_required_fields_returns_error(self):
        specialist = _SchemaSpecialist()
        # JSON parses fine but the required `result` field is missing.
        result = specialist._parse_and_validate_output(
            self._make_response('{"other": "x"}'),
            LLMUsage(),
            [],
        )
        assert result["success"] is False
        assert "Missing required fields" in result["error"]
        assert "result" in result["error"]

    def test_pydantic_validation_success(self):
        specialist = _PydanticSpecialist()
        result = specialist._parse_and_validate_output(
            self._make_response('{"result": "ok"}'),
            LLMUsage(),
            [],
        )
        assert result["success"] is True
        assert result["output"] == {"result": "ok"}

    def test_pydantic_validation_failure(self):
        specialist = _PydanticSpecialist()
        # pydantic_model requires `result` — missing it must fail validation.
        result = specialist._parse_and_validate_output(
            self._make_response('{"other": "x"}'),
            LLMUsage(),
            [],
        )
        assert result["success"] is False
        assert "schema validation failed" in result["error"]

    def test_clean_json_fences_then_parses(self):
        """End-to-end: a fenced JSON payload must be cleaned AND parsed by
        _parse_and_validate_output — the helper is actually wired in."""
        specialist = _PydanticSpecialist()
        result = specialist._parse_and_validate_output(
            self._make_response('```json\n{"result": "ok"}\n```'),
            LLMUsage(),
            [],
        )
        assert result["success"] is True
        assert result["output"] == {"result": "ok"}

    def test_raw_content_returned_when_no_schema_expected(self):
        """When neither output_schema nor pydantic_model is set, the raw
        content is returned under the `raw` key (no validation failure)."""

        class _NoSchemaSpecialist(Specialist):
            config: ClassVar[SpecialistConfig] = SpecialistConfig(
                name="@test-no-schema",
                description="no schema specialist for tests",
                system_prompt="You are a test specialist.",
                tools=[],
                max_iterations=1,
                default_model="mock/test",
            )

        specialist = _NoSchemaSpecialist()
        result = specialist._parse_and_validate_output(
            self._make_response("just plain text, not JSON"),
            LLMUsage(),
            [],
        )
        assert result["success"] is True
        assert result["output"] == {"raw": "just plain text, not JSON"}


# ============================================================
# 5. Bonus: LiteLLMProvider.peek_cost
# ============================================================


class TestLiteLLMPeekCost:
    def test_peek_cost_returns_non_none_for_known_model(self):
        # LiteLLMProvider.__init__ requires litellm to be importable; it is
        # listed as a hard dependency in pyproject.toml.
        from arnes.llm.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider()
        messages = [LLMMessage(role="user", content="a" * 4000)]  # ~1000 tokens
        cost = provider.peek_cost(
            model="anthropic/claude-sonnet-4-20250514",
            messages=messages,
        )
        assert cost is not None
        # $3.00/1M input tokens * 1000 tokens = $0.003
        assert cost == pytest.approx(0.003, rel=0.05)

    def test_peek_cost_returns_non_none_for_unknown_model(self):
        from arnes.llm.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider()
        messages = [LLMMessage(role="user", content="a" * 4000)]
        cost = provider.peek_cost(model="unknown/model", messages=messages)
        # Fallback: $1/1M tokens * 1000 tokens = $0.001
        assert cost is not None
        assert cost == pytest.approx(0.001, rel=0.05)

    def test_peek_cost_zero_for_empty_messages(self):
        from arnes.llm.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider()
        cost = provider.peek_cost(
            model="anthropic/claude-sonnet-4-20250514",
            messages=[],
        )
        assert cost == 0.0


# ============================================================
# 6. Bonus: @reviewer pydantic_model
# ============================================================


class TestReviewerPydanticModel:
    def test_reviewer_config_has_pydantic_model(self):
        assert Reviewer.config.pydantic_model is ReviewerOutput

    def test_reviewer_pydantic_model_validates_good_payload(self):
        output = ReviewerOutput.model_validate(
            {
                "verdict": "approve",
                "issues": [
                    {
                        "severity": "minor",
                        "file": "x.py",
                        "line": 1,
                        "issue": "typo",
                        "suggestion": "fix",
                    }
                ],
                "summary": "LGTM",
            }
        )
        assert output.verdict == "approve"
        assert len(output.issues) == 1

    def test_reviewer_pydantic_model_rejects_bad_verdict(self):
        """pydantic_model catches type/enum errors that JSON-schema `required`
        check would miss — that's the whole point of plumbing it through."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ReviewerOutput.model_validate(
                {"verdict": "ok", "issues": [], "summary": "x"}  # 'ok' is not a valid verdict
            )

    def test_reviewer_pydantic_model_is_baseModel_subclass(self):
        assert issubclass(ReviewerOutput, BaseModel)
