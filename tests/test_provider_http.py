"""Regression tests: Ollama and Anthropic clients against mocked HTTP servers."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import argparse

from openai import OpenAI

from self_correct import AnthropicMessagesClient, AntiHallucinator
from self_correct.cli import _build_client


def _args(**overrides):
    defaults = dict(provider="openai", base_url=None, api_key_env=None, timeout=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)

PIPELINE_TEXTS = [
    "Moon is cheese. Water boils at 100C.",
    "1. Moon is cheese.\n2. Water boils at 100C.",
    "VERIFIED: False. The moon is not cheese.",
    "VERIFIED: True.",
    "Moon is rock. Water boils at 100C.",
]


def _openai_body(text: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1,
            "model": "llama3.2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
    ).encode("utf-8")


def _anthropic_body(text: str) -> bytes:
    return json.dumps(
        {
            "id": "msg_mock",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }
    ).encode("utf-8")


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


class _RecordingHandler(BaseHTTPRequestHandler):
    """Serve canned replies and record request path/body/headers."""

    replies: List[bytes] = []
    requests: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:  # noqa: N802
        payload = _read_json(self)
        with self.lock:
            self.requests.append(
                {
                    "path": self.path,
                    "body": payload,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                }
            )
            body = self.replies.pop(0) if self.replies else b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _handler(bodies: List[bytes]):
    class Handler(_RecordingHandler):
        replies = list(bodies)
        requests: List[Dict[str, Any]] = []
        lock = threading.Lock()

    return Handler


def _run_pipeline(client) -> Any:
    agent = AntiHallucinator(client=client, strictness=1.0)
    return agent.generate(
        model="mock-model",
        prompt="Write two short facts about the moon and water.",
    )


def test_ollama_openai_compatible_base_url_runs_pipeline_against_mock():
    handler = _handler([_openai_body(text) for text in PIPELINE_TEXTS])
    with _serve(handler) as server:
        host, port = server.server_address
        client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="ollama",
        )
        result = _run_pipeline(client)

    assert "Moon is rock" in result.content
    assert len(result.hallucinations_caught) == 1
    assert result.token_usage.prompt_tokens == 50
    assert result.token_usage.completion_tokens == 100
    paths = [item["path"] for item in handler.requests]
    assert paths == ["/v1/chat/completions"] * 5
    first = handler.requests[0]["body"]
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][1]["role"] == "user"


def test_ollama_cli_client_uses_mocked_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    handler = _handler([_openai_body("passthrough draft")])
    with _serve(handler) as server:
        host, port = server.server_address
        client = _build_client(
            _args(provider="ollama", base_url=f"http://{host}:{port}/v1")
        )
        agent = AntiHallucinator(client=client, strictness=0.0)
        result = agent.generate(model="llama3.2", prompt="Hello")

    assert result.content == "passthrough draft"
    assert handler.requests[0]["path"] == "/v1/chat/completions"


def test_anthropic_messages_base_url_runs_pipeline_against_mock():
    handler = _handler([_anthropic_body(text) for text in PIPELINE_TEXTS])
    with _serve(handler) as server:
        host, port = server.server_address
        client = AnthropicMessagesClient(
            api_key="sk-ant-test",
            base_url=f"http://{host}:{port}/v1",
        )
        result = _run_pipeline(client)

    assert "Moon is rock" in result.content
    assert len(result.hallucinations_caught) == 1
    assert result.token_usage.prompt_tokens == 60
    assert result.token_usage.completion_tokens == 40
    assert [item["path"] for item in handler.requests] == ["/v1/messages"] * 5
    first = handler.requests[0]
    assert first["headers"]["x-api-key"] == "sk-ant-test"
    assert first["headers"]["anthropic-version"] == "2023-06-01"
    assert first["body"]["system"]
    assert first["body"]["messages"][0]["role"] == "user"
    assert "system" not in {m["role"] for m in first["body"]["messages"]}
    assert first["body"]["max_tokens"] == 4096


def test_anthropic_cli_client_uses_mocked_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-cli")
    handler = _handler([_anthropic_body("cli draft")])
    with _serve(handler) as server:
        host, port = server.server_address
        client = _build_client(
            _args(provider="anthropic", base_url=f"http://{host}:{port}")
        )
        agent = AntiHallucinator(client=client, strictness=0.0)
        result = agent.generate(model="claude-sonnet-4-5", prompt="Hello")

    assert result.content == "cli draft"
    assert handler.requests[0]["path"] == "/v1/messages"
    assert handler.requests[0]["headers"]["x-api-key"] == "sk-ant-cli"


def test_anthropic_structured_output_falls_back_on_mocked_messages_api():
    """JSON-mode kwargs are rejected; the run still completes as free text."""

    handler = _handler([_anthropic_body(text) for text in PIPELINE_TEXTS])
    with _serve(handler) as server:
        host, port = server.server_address
        client = AnthropicMessagesClient(
            api_key="sk-ant-test",
            base_url=f"http://{host}:{port}/v1",
        )
        agent = AntiHallucinator(
            client=client, strictness=1.0, structured_output="json"
        )
        result = agent.generate(
            model="claude-sonnet-4-5",
            prompt="Write two short facts about the moon and water.",
        )

    assert "Moon is rock" in result.content
    # Draft has no extra kwargs; extract/verify retry after TypeError.
    assert len(handler.requests) == 5
    for item in handler.requests:
        assert "response_format" not in item["body"]
