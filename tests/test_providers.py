"""Tests for provider selection."""

import argparse

import pytest

from self_correct import cli


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
