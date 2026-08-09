"""LiteLLM-based provider for paid vendors (Anthropic, OpenAI, Google, Groq, etc.)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from arnes.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMUsage

# Cost per 1M tokens (USD) — kept up to date as of 2026-01. Used for cost guard.
# Source: official pricing pages. Update when vendors change pricing.
_PRICING_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "anthropic/claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "anthropic/claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/o1": {"input": 15.00, "output": 60.00},
    "openai/o1-mini": {"input": 3.00, "output": 12.00},
    "google/gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "google/gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "groq/llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate cost in USD for a single LLM call."""
    pricing = _PRICING_USD_PER_1M_TOKENS.get(model)
    if not pricing:
        # Fallback: assume $1/1M tokens (conservative)
        return (tokens_in + tokens_out) * 1.0 / 1_000_000
    return (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1_000_000


def _estimate_input_tokens(messages: list[LLMMessage]) -> int:
    """Cheap local estimate of input tokens.

    4 chars ≈ 1 token is a deliberately rough heuristic — good enough for a
    pre-flight budget check, and avoids pulling in a per-vendor tokenizer
    just to *estimate* a call we haven't made yet.
    """
    return sum(len(m.content) // 4 for m in messages)


def _get_delta_content(delta: Any) -> str:
    """Extract ``content`` from a streaming chunk's delta.

    ``delta`` may be a litellm ``Delta`` pydantic model OR a plain dict —
    litellm's ``ModelResponse`` constructor converts ``StreamingChoices``
    into ``Choices`` and serializes ``delta`` to a dict in the process.
    This helper handles both shapes so the provider works against real
    litellm streams and against test-constructed ``ModelResponse`` objects.
    """
    if delta is None:
        return ""
    if isinstance(delta, dict):
        return delta.get("content") or ""
    return getattr(delta, "content", None) or ""


def _get_usage_field(usage: Any, field: str) -> int:
    """Extract a token-count field from a usage object or dict.

    Same dict-or-object reasoning as :func:`_get_delta_content` — litellm
    may serialize ``usage`` to a dict depending on the code path.
    """
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return usage.get(field, 0) or 0
    return getattr(usage, field, 0) or 0


def _merge_delta_tool_calls(delta: Any, acc: list[dict[str, Any]]) -> None:
    """Merge one streamed ``delta.tool_calls`` fragment into ``acc``.

    ``delta`` may be a litellm ``Delta`` pydantic model OR a plain dict.
    Streaming providers send the tool call as fragments across several
    chunks: the first chunk carries the call's ``index``/``id``/``name``,
    later chunks append ``arguments`` pieces. Entries in ``acc`` are keyed
    by ``index``; string arguments are concatenated in arrival order so a
    split JSON-arguments string reassembles correctly. Non-string argument
    fragments (rare) are JSON-serialized on the spot.
    """
    raw = delta.get("tool_calls") if isinstance(delta, dict) else getattr(delta, "tool_calls", None)
    if not raw:
        return
    for item in raw:
        if isinstance(item, dict):
            index = item.get("index")
            function = item.get("function")
            call_id = item.get("id")
        else:
            index = getattr(item, "index", None)
            function = getattr(item, "function", None)
            call_id = getattr(item, "id", None)
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")
        elif function is not None:
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
        else:
            name, arguments = None, None
        if arguments is not None and not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        entry = next((e for e in acc if e["index"] == index), None)
        if entry is None:
            entry = {"index": index, "id": call_id, "name": name, "args_parts": []}
            acc.append(entry)
        if not entry["name"] and name:
            entry["name"] = name
        if not entry["id"] and call_id:
            entry["id"] = call_id
        if arguments:
            entry["args_parts"].append(arguments)


def _finalize_tool_calls(acc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize accumulated stream fragments to the OpenAI tool_calls shape."""
    tool_calls: list[dict[str, Any]] = []
    for entry in acc:
        name = entry["name"]
        if not name:
            continue
        arguments = "".join(entry["args_parts"]) if entry["args_parts"] else "{}"
        tool_calls.append(
            {
                "id": entry["id"] or f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return tool_calls


class LiteLLMProvider(LLMProvider):
    """Universal provider for paid vendors via LiteLLM.

    Reads API keys from environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.).
    NEVER stores or logs the keys.

    Accepts arbitrary ``**kwargs`` at construction time (e.g. ``api_key``,
    ``base_url``, ``timeout``) and forwards them to every ``litellm.acompletion``
    call. Caller-supplied kwargs at ``complete()`` time take precedence over
    construction-time kwargs — this lets a test/dev override (e.g. an explicit
    ``api_key="sk-test"``) reach litellm without being silently dropped, while
    still allowing the default env-var-based resolution when no kwargs are
    given.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Validate that LiteLLM is available
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LiteLLM is required for paid providers. Install with: pip install litellm"
            ) from e

        # Store construction-time kwargs (api_key, base_url, etc.) so they can
        # be merged into every litellm.acompletion call. We intentionally do
        # NOT introspect or persist anything that looks like a secret —
        # litellm handles env-var resolution and key rotation itself.
        self._init_kwargs: dict[str, Any] = dict(kwargs)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,  # Accepted but ignored
        **kwargs: Any,
    ) -> LLMResponse:
        import litellm

        # LiteLLM uses "provider/model" format directly
        litellm_messages = [m.model_dump(exclude_none=True) for m in messages]

        # Build the call kwargs WITHOUT clobbering caller-supplied **kwargs.
        # Order of precedence (lowest → highest):
        #   1. construction-time kwargs (self._init_kwargs)
        #   2. per-call kwargs (passed to complete())
        #   3. explicit named params (model, temperature, max_tokens, ...)
        # Step 3 wins because it's applied LAST via call_kwargs.update below.
        call_kwargs: dict[str, Any] = {**self._init_kwargs}
        call_kwargs.update(kwargs)
        call_kwargs["model"] = model
        call_kwargs["messages"] = litellm_messages
        call_kwargs["temperature"] = temperature
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens
        if tools:
            call_kwargs["tools"] = tools
        if response_format and response_format.get("type") == "json_object":
            call_kwargs["response_format"] = {"type": "json_object"}
        # Note: per-call kwargs were already merged above (step 2), so any
        # caller-supplied `top_p`, `seed`, `user`, etc. reach litellm.

        # LiteLLM async call
        response = await litellm.acompletion(**call_kwargs)

        # Extract standard fields
        content = response.choices[0].message.content or ""
        tool_calls = []
        if (
            hasattr(response.choices[0].message, "tool_calls")
            and response.choices[0].message.tool_calls
        ):
            for tc in response.choices[0].message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        # ``response.usage`` may be unset (litellm constructs ModelResponse
        # without populating ``usage`` for cached / short-circuited responses
        # — the attribute is absent, not None). ``getattr(..., None)`` keeps
        # the ``if usage else 0`` fallback below actually effective instead
        # of letting an AttributeError escape to the caller.
        usage = getattr(response, "usage", None)
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        cost = _estimate_cost(model, tokens_in, tokens_out)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=LLMUsage(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                model=model,
                cached=False,
            ),
            model=model,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def list_models(self) -> list[str]:
        return list(_PRICING_USD_PER_1M_TOKENS.keys())

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,  # Accepted but ignored
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Stream a completion via ``litellm.acompletion(stream=True)``.

        LiteLLM returns a ``CustomStreamWrapper`` async iterator. Each
        chunk has ``choices[0].delta.content`` with the new token. Usage
        stats are not always streamed by vendors — when litellm surfaces
        them on a chunk, we capture them and yield a final ``LLMResponse``
        with the full ``LLMUsage`` after the stream ends.

        Contract:

        - Each chunk with non-empty ``delta.content`` yields an
          ``LLMResponse`` whose ``content`` is *just the new token*.
        - Intermediate chunks carry an empty ``LLMUsage`` (zeros).
        - After the stream ends, if any chunk carried usage, a final
          ``LLMResponse`` is yielded with the full ``LLMUsage`` (tokens
          + cost). If no usage was streamed, no final usage chunk is
          yielded — callers detect this by the absence of a chunk with
          non-zero ``usage.tokens_out``.
        """
        import litellm

        litellm_messages = [m.model_dump(exclude_none=True) for m in messages]

        # Build the call kwargs with the same precedence as complete():
        # init kwargs < per-call kwargs < explicit named params.
        call_kwargs: dict[str, Any] = {**self._init_kwargs}
        call_kwargs.update(kwargs)
        call_kwargs["model"] = model
        call_kwargs["messages"] = litellm_messages
        call_kwargs["temperature"] = temperature
        call_kwargs["stream"] = True
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens
        if tools:
            call_kwargs["tools"] = tools
        if response_format and response_format.get("type") == "json_object":
            call_kwargs["response_format"] = {"type": "json_object"}

        # ``acompletion`` is async and returns a ``CustomStreamWrapper``
        # when ``stream=True``. The wrapper is an async iterator.
        stream = await litellm.acompletion(**call_kwargs)

        # Track the last usage seen — vendors that stream usage send it on
        # the final chunk; vendors that don't (OpenAI without
        # ``stream_options={"include_usage": True}``) never send it.
        final_tokens_in = 0
        final_tokens_out = 0
        final_cost = 0.0
        saw_usage = False
        # Streaming tool_calls arrive as fragments spread over multiple
        # chunks (index/id/name first, then arguments pieces). Accumulate
        # and re-emit as a single assembled tool_calls chunk at the end.
        pending_tool_calls: list[dict[str, Any]] = []

        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                # Some vendors send a final chunk with only ``usage`` and
                # an empty ``choices`` list — capture usage from it.
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    final_tokens_in = _get_usage_field(usage, "prompt_tokens")
                    final_tokens_out = _get_usage_field(usage, "completion_tokens")
                    final_cost = _estimate_cost(model, final_tokens_in, final_tokens_out)
                    saw_usage = True
                continue

            delta = getattr(choices[0], "delta", None)
            # ``delta`` may be a ``Delta`` pydantic model OR a plain dict —
            # litellm's ``ModelResponse`` constructor converts
            # ``StreamingChoices`` into ``Choices`` and serializes ``delta``
            # to a dict in the process. Handle both so the provider works
            # against real litellm streams and against test-constructed
            # ``ModelResponse`` objects.
            content = _get_delta_content(delta)

            # Fold any streamed tool_call fragments into the accumulator.
            _merge_delta_tool_calls(delta, pending_tool_calls)

            # Capture usage if present on this chunk (litellm surfaces it
            # on the final chunk for vendors that stream usage).
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                ti = _get_usage_field(usage, "prompt_tokens")
                to = _get_usage_field(usage, "completion_tokens")
                if ti or to:
                    final_tokens_in = ti
                    final_tokens_out = to
                    final_cost = _estimate_cost(model, ti, to)
                    saw_usage = True

            if content:
                yield LLMResponse(
                    content=content,
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

        # Assembled tool calls are yielded as one chunk so the consumer's
        # last-non-empty-wins accumulation picks them up even though no
        # single streamed chunk ever carried the complete list.
        if pending_tool_calls:
            assembled = _finalize_tool_calls(pending_tool_calls)
            if assembled:
                yield LLMResponse(
                    content="",
                    tool_calls=assembled,
                    usage=LLMUsage(
                        tokens_in=0,
                        tokens_out=0,
                        cost_usd=0.0,
                        model=model,
                        cached=False,
                    ),
                    model=model,
                )

        # Yield a final chunk with usage stats if we collected them.
        if saw_usage:
            yield LLMResponse(
                content="",
                tool_calls=[],
                usage=LLMUsage(
                    tokens_in=final_tokens_in,
                    tokens_out=final_tokens_out,
                    cost_usd=final_cost,
                    model=model,
                    cached=False,
                ),
                model=model,
            )

    def peek_cost(
        self,
        *,
        model: str,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> float | None:
        """Pre-flight cost estimate based on input tokens only.

        We can't know the output token count until the model actually
        generates a response, so we conservatively estimate *just* the input
        portion of the bill. CostGuard uses this to reject calls that would
        breach the budget *before* any money is spent — without this, the
        pre-flight check in ``CostGuard`` is dead code for LiteLLMProvider.

        The estimate is intentionally a lower bound (output cost is unknown
        and not added), which is the safe direction for a budget guard: we
        never *over*-estimate and thereby block legitimate calls.
        """
        input_tokens = _estimate_input_tokens(messages)
        pricing = _PRICING_USD_PER_1M_TOKENS.get(model)
        if not pricing:
            # Fallback: $1/1M tokens (matches _estimate_cost's fallback)
            return input_tokens * 1.0 / 1_000_000
        return input_tokens * pricing["input"] / 1_000_000
