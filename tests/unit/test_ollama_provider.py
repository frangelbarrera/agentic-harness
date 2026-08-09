"""Tests for the Ollama provider (arnes/llm/ollama.py).

Drives the real provider against a local HTTP server that emulates
Ollama's /api/chat endpoints (stream + non-stream), plus connection
failure cases. No outside network required.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from arnes.llm.base import LLMMessage
from arnes.llm.ollama import OllamaProvider


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """Serves canned /api/chat responses scripted via class vars."""

    protocol_version = "HTTP/1.0"
    ndjson_lines: ClassVar[list[str]] = []
    requests_seen: ClassVar[list[dict[str, object]]] = []

    def _read_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        return json.loads(body) if body else {}

    def do_POST(self):
        payload = self._read_body()
        self.__class__.requests_seen.append(payload)
        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for line in self.__class__.ndjson_lines:
            self.wfile.write(line.encode())
            self.wfile.flush()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _FakeOllamaHandler.requests_seen = []
    httpd = HTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()


def _next_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _provider(server) -> OllamaProvider:
    port = server.server_address[1]
    return OllamaProvider(host=f"http://127.0.0.1:{port}", timeout=10.0)


def _m(msg: str) -> list[LLMMessage]:
    return [LLMMessage(role="user", content=msg)]


async def test_complete_returns_content_and_usage(server):
    _FakeOllamaHandler.ndjson_lines = [
        json.dumps(
            {
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "hello"},
                "done": True,
                "prompt_eval_count": 12,
                "eval_count": 3,
            }
        )
    ]
    p = _provider(server)
    resp = await p.complete(_m("hi"), model="ollama/llama3.2")
    assert resp.content == "hello"
    assert resp.usage.tokens_in == 12
    assert resp.usage.tokens_out == 3
    assert resp.usage.cost_usd == 0.0
    assert resp.model == "ollama/llama3.2"


async def test_complete_normalizes_tool_calls(server):
    _FakeOllamaHandler.ndjson_lines = [
        json.dumps(
            {
                "model": "llama3.2",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "filesystem_read", "arguments": {"path": "a.txt"}}},
                        {"function": {"name": "empty_args", "arguments": None}},
                        {"function": {"arguments": {"x": 1}}},
                        "garbage",
                    ],
                },
                "done": True,
            }
        )
    ]
    p = _provider(server)
    resp = await p.complete(_m("read"), model="ollama/llama3.2")
    assert len(resp.tool_calls) == 2
    first = resp.tool_calls[0]
    assert first["type"] == "function"
    assert first["function"]["name"] == "filesystem_read"
    assert json.loads(first["function"]["arguments"]) == {"path": "a.txt"}
    assert resp.tool_calls[1]["function"]["arguments"] == "{}"


async def test_complete_raises_on_connect_error(server):
    p = OllamaProvider(host=f"http://127.0.0.1:{_next_free_port()}", timeout=2.0)
    with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
        await p.complete(_m("hi"), model="llama3.2")


async def test_stream_propagates_tool_calls(server):
    """Regression: tool_calls arriving over NDJSON were previously dropped."""
    _FakeOllamaHandler.ndjson_lines = [
        json.dumps({"message": {"content": ""}, "done": False}) + "\n",
        json.dumps(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "filesystem_read", "arguments": {"path": "a.txt"}}}
                    ],
                },
                "done": False,
            }
        )
        + "\n",
        json.dumps({"message": {"content": ""}, "done": True,
                    "prompt_eval_count": 100, "eval_count": 20})
        + "\n",
    ]
    p = _provider(server)
    chunks = [c async for c in p.stream_complete(_m("go"), model="ollama/llama3.2")]
    assert len(chunks) == 3
    calls = [c.tool_calls for c in chunks if c.tool_calls]
    assert len(calls) == 1
    assert calls[0][0]["function"]["name"] == "filesystem_read"


async def test_stream_yields_usage_on_final_chunk(server):
    _FakeOllamaHandler.ndjson_lines = [
        json.dumps({"message": {"content": "he"}, "done": False}) + "\n",
        json.dumps({"message": {"content": "llo"}, "done": False}) + "\n",
        json.dumps({"message": {"content": ""}, "done": True,
                     "prompt_eval_count": 40, "eval_count": 5})
        + "\n",
    ]
    p = _provider(server)
    chunks = [c async for c in p.stream_complete(_m("hi"), model="ollama/llama3.2")]
    assert "".join(c.content for c in chunks) == "hello"
    final = chunks[-1]
    assert final.usage.tokens_in == 40
    assert final.usage.tokens_out == 5


async def test_stream_skips_malformed_lines(server):
    _FakeOllamaHandler.ndjson_lines = [
        "this is not json\n",
        json.dumps({"message": {"content": "ok"}, "done": False}) + "\n",
        json.dumps({"message": {"content": ""}, "done": True,
                     "prompt_eval_count": 1, "eval_count": 1})
        + "\n",
    ]
    p = _provider(server)
    chunks = [c async for c in p.stream_complete(_m("hi"), model="ollama/llama3.2")]
    assert "".join(c.content for c in chunks) == "ok"


async def test_stream_sentinel_when_done_missing(server):
    """Server closes mid-stream without a done line: sentinel usage chunk."""
    _FakeOllamaHandler.ndjson_lines = [
        json.dumps({"message": {"content": "partial"}, "done": False}) + "\n",
    ]
    p = _provider(server)
    chunks = [c async for c in p.stream_complete(_m("hi"), model="ollama/llama3.2")]
    assert [c.content for c in chunks] == ["partial", ""]
    assert chunks[-1].usage.tokens_in == 0
    assert chunks[-1].usage.tokens_out == 0


async def test_complete_skips_malformed_function_tool_call(server):
    """A tool_call whose ``function`` is not a dict must be skipped, not crash."""
    _FakeOllamaHandler.ndjson_lines = [
        json.dumps(
            {
                "model": "llama3.2",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "filesystem_read", "arguments": {"p": 1}}},
                        "function-is-a-string",
                        {"function": 42},
                        {"function": {"name": "", "arguments": {}}},
                    ],
                },
                "done": True,
            }
        )
    ]
    p = _provider(server)
    resp = await p.complete(_m("read"), model="ollama/llama3.2")
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["function"]["name"] == "filesystem_read"


async def test_stream_accumulates_tool_calls_across_chunks(server):
    """tool_calls split across several NDJSON lines must not be lost.

    Ollama usually delivers the full list on one line, but a vendor that
    splits the call across chunks must still surface the complete set to
    the consumer (which uses last-non-empty-wins).
    """
    _FakeOllamaHandler.ndjson_lines = [
        json.dumps(
            {"message": {"tool_calls": [{"function": {"name": "tool_a", "arguments": {"x": 1}}}]},
             "done": False}
        )
        + "\n",
        json.dumps(
            {"message": {"tool_calls": [{"function": {"name": "tool_b", "arguments": {"y": 2}}}]},
             "done": False}
        )
        + "\n",
        json.dumps({"message": {"content": ""}, "done": True,
                     "prompt_eval_count": 10, "eval_count": 2})
        + "\n",
    ]
    p = _provider(server)
    chunks = [c async for c in p.stream_complete(_m("go"), model="ollama/llama3.2")]
    tool_call_chunks = [c.tool_calls for c in chunks if c.tool_calls]
    assert len(tool_call_chunks) >= 1
    names = [tc["function"]["name"] for tc in tool_call_chunks[-1]]
    assert names == ["tool_a", "tool_b"]


async def test_stream_no_duplicates_when_done_repeats_tool_calls(server):
    """If the done line repeats the tool_calls list, ids are deduplicated."""
    _FakeOllamaHandler.ndjson_lines = [
        json.dumps(
            {"message": {"tool_calls": [{"id": "call_1", "function": {"name": "tool_a",
                                                                        "arguments": {}}}]},
             "done": False}
        )
        + "\n",
        json.dumps(
            {"message": {"tool_calls": [{"id": "call_1", "function": {"name": "tool_a",
                                                                       "arguments": {}}}]},
             "done": True, "prompt_eval_count": 5, "eval_count": 1}
        )
        + "\n",
    ]
    p = _provider(server)
    chunks = [c async for c in p.stream_complete(_m("go"), model="ollama/llama3.2")]
    tool_call_chunks = [c.tool_calls for c in chunks if c.tool_calls]
    assert tool_call_chunks
    # The repeated call never accumulates into a longer list: every emitted
    # chunk carries exactly one, and always the same id.
    assert all(len(tc) == 1 for tc in tool_call_chunks)
    assert {tc[0]["id"] for tc in tool_call_chunks} == {"call_1"}


async def test_stream_wraps_connection_failure():
    """Both ConnectError and ConnectTimeout surface as the friendly RuntimeError."""
    p = OllamaProvider(host=f"http://127.0.0.1:{_next_free_port()}", timeout=2.0)

    async def collect() -> None:
        async for _ in p.stream_complete(_m("hi"), model="llama3.2"):
            pass

    with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
        await collect()
