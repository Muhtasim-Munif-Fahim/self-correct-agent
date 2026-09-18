"""Tests for provider selection and the Anthropic Messages adapter."""

import argparse
from types import SimpleNamespace

import pytest

from self_correct import AnthropicMessagesClient, cli
from self_correct.providers import (
    build_client,
    build_messages_payload,
    completion_from_anthropic,
    content_to_text,
    messages_endpoint,
    openai_messages_to_anthropic,
)


def _args(**overrides):
    defaults = dict(provider="openai", base_url=None, api_key_env=None, timeout=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildClient:
    def test_openai_uses_the_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = cli._build_client(_args())
        assert "openai.com" in str(client.base_url)

    def test_ollama_defaults_to_the_local_endpoint(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = cli._build_client(_args(provider="ollama"))
        assert "11434" in str(client.base_url)

    def test_ollama_needs_no_api_key(self, monkeypatch):
        """A local server ignores the key, but the client insists on one."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert cli._build_client(_args(provider="ollama")) is not None

    def test_custom_requires_a_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with pytest.raises(SystemExit, match="requires --base-url"):
            cli._build_client(_args(provider="custom"))

    def test_custom_uses_the_given_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = cli._build_client(
            _args(provider="custom", base_url="http://localhost:9999/v1")
        )
        assert "localhost:9999" in str(client.base_url)

    def test_base_url_overrides_the_provider_default(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = cli._build_client(
            _args(provider="ollama", base_url="http://elsewhere:1234/v1")
        )
        assert "elsewhere:1234" in str(client.base_url)

    def test_api_key_env_names_a_different_variable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MY_KEY", "sk-other")
        assert cli._build_client(_args(api_key_env="MY_KEY")) is not None

    def test_missing_key_names_the_variable_it_looked_for(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
            cli._build_client(_args())

    def test_anthropic_uses_the_messages_adapter(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        client = cli._build_client(_args(provider="anthropic"))
        assert isinstance(client, AnthropicMessagesClient)
        assert "anthropic.com" in client.base_url
        assert client.api_key == "sk-ant-test"

    def test_anthropic_ignores_openai_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
        client = cli._build_client(_args(provider="anthropic"))
        assert client.api_key == "sk-ant-real"

    def test_anthropic_missing_key_names_anthropic_variable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
            cli._build_client(_args(provider="anthropic"))

    def test_anthropic_api_key_env_override(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("TEAM_ANTHROPIC_KEY", "sk-ant-team")
        client = cli._build_client(
            _args(provider="anthropic", api_key_env="TEAM_ANTHROPIC_KEY")
        )
        assert client.api_key == "sk-ant-team"

    def test_anthropic_base_url_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        client = cli._build_client(
            _args(provider="anthropic", base_url="http://proxy.local/v1")
        )
        assert client.base_url == "http://proxy.local/v1"


class TestAnthropicAdapter:
    def test_system_messages_are_split_out(self):
        system, messages = openai_messages_to_anthropic(
            [
                {"role": "system", "content": "Be careful."},
                {"role": "user", "content": "Hello"},
            ]
        )
        assert system == "Be careful."
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_content_blocks_flatten_to_text(self):
        assert content_to_text(
            [{"type": "text", "text": "Moon "}, {"type": "text", "text": "is rock."}]
        ) == "Moon is rock."

    def test_messages_endpoint_accepts_versioned_and_root_urls(self):
        assert messages_endpoint("https://api.anthropic.com/v1") == (
            "https://api.anthropic.com/v1/messages"
        )
        assert messages_endpoint("https://api.anthropic.com") == (
            "https://api.anthropic.com/v1/messages"
        )

    def test_payload_requires_max_tokens(self):
        payload = build_messages_payload(
            "claude-sonnet-4-5",
            [{"role": "user", "content": "Hi"}],
            temperature=0.2,
            max_tokens=None,
        )
        assert payload["max_tokens"] == 4096
        assert payload["messages"][0]["role"] == "user"

    def test_completion_maps_usage_and_text(self):
        completion = completion_from_anthropic(
            {
                "id": "msg_1",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }
        )
        assert completion.choices[0].message.content == "Hello"
        assert completion.usage.prompt_tokens == 11
        assert completion.usage.completion_tokens == 7
        assert completion.usage.total_tokens == 18

    def test_create_posts_messages_and_returns_openai_shape(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = dict(headers)
            captured["timeout"] = timeout
            return {
                "id": "msg_1",
                "content": [{"type": "text", "text": "adapted"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }

        client = AnthropicMessagesClient(
            api_key="sk-ant-test",
            base_url="http://proxy.local/v1",
            timeout=12.5,
            transport=transport,
        )
        response = client.chat.completions.create(
            model="claude-sonnet-4-5",
            messages=[
                {"role": "system", "content": "Stay brief."},
                {"role": "user", "content": "Ping"},
            ],
            temperature=0.2,
            max_tokens=64,
        )
        assert captured["url"] == "http://proxy.local/v1/messages"
        assert captured["payload"]["system"] == "Stay brief."
        assert captured["payload"]["messages"] == [{"role": "user", "content": "Ping"}]
        assert captured["payload"]["max_tokens"] == 64
        assert captured["headers"]["x-api-key"] == "sk-ant-test"
        assert captured["headers"]["anthropic-version"] == "2023-06-01"
        assert captured["timeout"] == 12.5
        assert response.choices[0].message.content == "adapted"
        assert response.usage.prompt_tokens == 4

    def test_structured_kwargs_are_rejected_as_unknown_parameters(self):
        client = AnthropicMessagesClient(api_key="sk-ant-test", transport=lambda *a: {})
        with pytest.raises(TypeError, match="response_format"):
            client.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
                response_format={"type": "json_object"},
            )

    def test_build_client_factory_returns_anthropic_adapter(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        client = build_client("anthropic")
        assert isinstance(client, AnthropicMessagesClient)


class TestFormatCost:
    def test_tiny_cost_is_not_reported_as_zero(self):
        """$0.0000 reads as free; it isn't."""
        from self_correct.core import TokenUsage

        usage = TokenUsage(prompt_tokens=10, completion_tokens=10)
        assert cli._format_cost(usage, "gpt-4o-mini") == "<$0.0001"

    def test_unknown_model_says_so(self):
        from self_correct.core import TokenUsage

        text = cli._format_cost(TokenUsage(prompt_tokens=10, completion_tokens=10), "llama-3")
        assert "unknown" in text


def test_batch_uses_provider_client(tmp_path, monkeypatch):
    """batch used to hardcode OpenAI(); it must honour --provider."""

    from self_correct.cli import _build_parser, cmd_batch

    captured = {}

    class FakeHallucinator:
        def __init__(self, **kwargs):
            captured["client"] = kwargs["client"]

        def generate(self, **kwargs):
            return SimpleNamespace(
                to_dict=lambda: {
                    "content": "ok",
                    "hallucinations_caught": [],
                }
            )

    input_path = tmp_path / "in.jsonl"
    input_path.write_text('{"id": "1", "prompt": "hello"}\n', encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr("self_correct.cli.AntiHallucinator", FakeHallucinator)
    args = _build_parser().parse_args(
        [
            "batch",
            "--input",
            str(input_path),
            "--provider",
            "anthropic",
            "--format",
            "jsonl",
        ]
    )
    cmd_batch(args)
    assert isinstance(captured["client"], AnthropicMessagesClient)
