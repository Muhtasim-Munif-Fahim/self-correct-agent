"""Tests for self_correct.core.AntiHallucinator."""

import asyncio
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
from self_correct.core import AntiHallucinator, TokenUsage, _ClaimCache


def _mock_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    """Helper to create a mock OpenAI-style response with usage stats."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


# ------------------------------------------------------------------
# TokenUsage
# ------------------------------------------------------------------

def test_token_usage_total() -> None:
    """Total tokens = prompt + completion."""
    u = TokenUsage(prompt_tokens=100, completion_tokens=50)
    assert u.total_tokens == 150


def test_token_usage_cost_estimate() -> None:
    """Cost estimate should use per-1k pricing."""
    u = TokenUsage(prompt_tokens=1000, completion_tokens=1000)
    cost = u.estimate_cost(prompt_cost_per_1k=0.01, completion_cost_per_1k=0.03)
    assert abs(cost - 0.04) < 1e-9


def test_token_usage_add_is_thread_safe() -> None:
    """Concurrent updates should not lose token counts."""
    usage = TokenUsage()

    def _bump() -> None:
        for _ in range(10_000):
            usage.add(prompt_tokens=1, completion_tokens=2)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: _bump(), range(8)))

    assert usage.prompt_tokens == 80_000
    assert usage.completion_tokens == 160_000
    assert usage.total_tokens == 240_000


# ------------------------------------------------------------------
# ClaimCache
# ------------------------------------------------------------------

def test_cache_put_and_get() -> None:
    """Cache should store and retrieve results."""
    cache = _ClaimCache(max_size=10)
    cache.put("The sky is blue", {"is_valid": True, "critique": "ok"})
    result = cache.get("The sky is blue")
    assert result is not None
    assert result["is_valid"] is True


def test_cache_miss() -> None:
    """Cache should return None for missing entries."""
    cache = _ClaimCache(max_size=10)
    assert cache.get("Unknown claim") is None


def test_cache_case_insensitive() -> None:
    """Cache keys should be case-insensitive."""
    cache = _ClaimCache(max_size=10)
    cache.put("The Sky Is Blue", {"is_valid": True})
    assert cache.get("the sky is blue") is not None


def test_cache_eviction() -> None:
    """Oldest entry should be evicted when cache is full."""
    cache = _ClaimCache(max_size=2)
    cache.put("Claim A", {"id": "a"})
    cache.put("Claim B", {"id": "b"})
    cache.put("Claim C", {"id": "c"})
    assert cache.size == 2
    assert cache.get("Claim A") is None  # Evicted
    assert cache.get("Claim B") is not None


def test_cache_clear() -> None:
    """clear() should empty the cache."""
    cache = _ClaimCache(max_size=10)
    cache.put("X", {"ok": True})
    cache.clear()
    assert cache.size == 0


def test_cache_management_helpers() -> None:
    """AntiHallucinator should expose basic cache controls."""
    agent = AntiHallucinator(MagicMock(), cache_size=2)

    assert agent.cache_size == 0
    assert agent.cache_max_size == 2

    agent.cache.put("Claim", {"is_valid": True})
    assert agent.cache_size == 1

    agent.clear_cache()
    assert agent.cache_size == 0


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def test_strictness_clamped_high() -> None:
    agent = AntiHallucinator(MagicMock(), strictness=2.0)
    assert agent.strictness == 1.0


def test_strictness_clamped_low() -> None:
    agent = AntiHallucinator(MagicMock(), strictness=-1.0)
    assert agent.strictness == 0.0


# ------------------------------------------------------------------
# Claim Parsing
# ------------------------------------------------------------------

def test_parse_claims_numbered_dot() -> None:
    agent = AntiHallucinator(MagicMock())
    assert agent._parse_claims("1. First\n2. Second") == ["First", "Second"]


def test_parse_claims_numbered_paren() -> None:
    agent = AntiHallucinator(MagicMock())
    assert agent._parse_claims("1) First\n2) Second") == ["First", "Second"]


def test_parse_claims_bullet_dash() -> None:
    agent = AntiHallucinator(MagicMock())
    assert agent._parse_claims("- First\n- Second") == ["First", "Second"]


def test_parse_claims_empty_input() -> None:
    agent = AntiHallucinator(MagicMock())
    assert agent._parse_claims("") == []


# ------------------------------------------------------------------
# None content handling
# ------------------------------------------------------------------

def test_call_llm_none_content() -> None:
    mock_client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = None
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 0
    mock_client.chat.completions.create.return_value = resp

    agent = AntiHallucinator(mock_client)
    result = agent._call_llm("dummy", "sys", "usr")
    assert result == ""


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

def test_call_llm_api_error_wrapped() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("timeout")
    agent = AntiHallucinator(mock_client)
    with pytest.raises(RuntimeError, match="LLM API call failed"):
        agent._call_llm("dummy", "sys", "usr")


def test_call_llm_bad_client_interface() -> None:
    agent = AntiHallucinator("not_a_client")
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        agent._call_llm("dummy", "sys", "usr")


# ------------------------------------------------------------------
# Strictness levels
# ------------------------------------------------------------------

def test_strictness_zero_bypasses() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response("Draft.")
    agent = AntiHallucinator(mock_client, strictness=0.0)
    result = agent.generate(model="dummy", prompt="Hello")
    assert mock_client.chat.completions.create.call_count == 1
    assert result.content == "Draft."


def test_strictness_medium_uses_light_critique() -> None:
    agent = AntiHallucinator(MagicMock(), strictness=0.5)
    prompt = agent._build_critique_prompt()
    assert "obvious factual errors" in prompt.lower()


def test_strictness_high_uses_strict_critique() -> None:
    agent = AntiHallucinator(MagicMock(), strictness=1.0)
    prompt = agent._build_critique_prompt()
    assert "empirical" in prompt.lower()


def test_custom_prompts_are_used() -> None:
    """Custom prompts should flow through all pipeline stages."""
    mock_client = MagicMock()

    responses = [
        _mock_response("Draft output."),
        _mock_response("1. Draft output.\n2. Moon is cheese."),
        _mock_response("VERIFIED: True."),
        _mock_response("VERIFIED: False. The moon is not cheese."),
        _mock_response("Draft output."),
    ]

    def mock_create(*args, **kwargs):
        return responses.pop(0)

    mock_client.chat.completions.create.side_effect = mock_create

    agent = AntiHallucinator(
        mock_client,
        strictness=1.0,
        draft_system_prompt="Draft mode",
        extraction_prompt="Extract claims only",
        critique_prompt="Check claims with care",
        correction_prompt="Rewrite the draft conservatively",
    )

    agent.generate(model="dummy", prompt="Tell me something.")

    calls = mock_client.chat.completions.create.call_args_list
    assert calls[0].kwargs["messages"][0]["content"] == "Draft mode"
    assert calls[1].kwargs["messages"][0]["content"] == "Extract claims only"
    assert calls[2].kwargs["messages"][0]["content"] == "Check claims with care"
    assert calls[3].kwargs["messages"][0]["content"] == "Check claims with care"
    assert calls[4].kwargs["messages"][0]["content"] == "Rewrite the draft conservatively"


# ------------------------------------------------------------------
# Token tracking
# ------------------------------------------------------------------

def test_token_usage_accumulated() -> None:
    """Token usage should be accumulated across all pipeline calls."""
    mock_client = MagicMock()

    responses = [
        _mock_response("Draft text.", prompt_tokens=50, completion_tokens=30),
        _mock_response("1. A factual claim here.", prompt_tokens=60, completion_tokens=20),
        _mock_response("VERIFIED: True.", prompt_tokens=40, completion_tokens=10),
    ]

    def mock_create(*args, **kwargs):
        return responses.pop(0)

    mock_client.chat.completions.create.side_effect = mock_create

    agent = AntiHallucinator(mock_client, strictness=1.0)
    result = agent.generate(model="dummy", prompt="Test")

    assert result.token_usage.prompt_tokens == 150   # 50 + 60 + 40
    assert result.token_usage.completion_tokens == 60  # 30 + 20 + 10
    assert result.token_usage.total_tokens == 210
    assert result.elapsed_seconds > 0


# ------------------------------------------------------------------
# Claim caching
# ------------------------------------------------------------------

def test_cached_claim_skips_llm_call() -> None:
    """If a claim is cached, it should not call the LLM again."""
    mock_client = MagicMock()

    responses = [
        _mock_response("Paris is the capital of France."),
        _mock_response("1. Paris is the capital of France."),
        _mock_response("VERIFIED: True."),
    ]

    def mock_create(*args, **kwargs):
        return responses.pop(0)

    mock_client.chat.completions.create.side_effect = mock_create

    agent = AntiHallucinator(mock_client, strictness=1.0)

    # First call: 3 LLM calls (draft + extract + critique)
    result1 = agent.generate(model="dummy", prompt="Capital of France?")
    assert mock_client.chat.completions.create.call_count == 3
    assert result1.verification_log[0]["cached"] is False

    # Reset mock but keep the cache
    mock_client.chat.completions.create.reset_mock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("Paris is the capital of France."),
        _mock_response("1. Paris is the capital of France."),
        # No third call needed — claim is cached!
    ]

    result2 = agent.generate(model="dummy", prompt="Capital of France?")
    # Only 2 calls: draft + extract. Critique was cached.
    assert mock_client.chat.completions.create.call_count == 2
    assert result2.verification_log[0]["cached"] is True


# ------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------

def test_full_pipeline_catches_hallucination() -> None:
    mock_client = MagicMock()

    responses = [
        _mock_response("France's capital is Paris. Moon is cheese."),
        _mock_response("1. France's capital is Paris.\n2. Moon is cheese."),
        _mock_response("VERIFIED: True."),
        _mock_response("VERIFIED: False. Moon is rock."),
        _mock_response("France's capital is Paris."),
    ]

    def mock_create(*args, **kwargs):
        return responses.pop(0)

    mock_client.chat.completions.create.side_effect = mock_create

    agent = AntiHallucinator(mock_client, strictness=1.0)
    result = agent.generate(model="dummy", prompt="Tell me stuff.")

    assert mock_client.chat.completions.create.call_count == 5
    assert result.content == "France's capital is Paris."
    assert len(result.hallucinations_caught) == 1


# ------------------------------------------------------------------
# Tool-assisted verification
# ------------------------------------------------------------------

def test_tools_used_at_high_strictness() -> None:
    from self_correct.tools import SearchResult

    mock_client = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "MockSearch"
    mock_tool.search.return_value = [
        SearchResult(title="Page", snippet="Paris is capital.", url="https://x.com")
    ]

    responses = [
        _mock_response("Capital is Paris."),
        _mock_response("1. Capital is Paris."),
        _mock_response("VERIFIED: True."),
    ]

    def mock_create(*args, **kwargs):
        return responses.pop(0)

    mock_client.chat.completions.create.side_effect = mock_create

    agent = AntiHallucinator(mock_client, strictness=1.0, tools=[mock_tool])
    result = agent.generate(model="dummy", prompt="Capital?")

    mock_tool.search.assert_called_once()
    assert result.verification_log[0]["evidence_used"] is True


def test_tools_not_used_at_low_strictness() -> None:
    mock_client = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "MockSearch"

    responses = [
        _mock_response("Capital is Paris."),
        _mock_response("1. Capital is Paris."),
        _mock_response("VERIFIED: True."),
    ]

    def mock_create(*args, **kwargs):
        return responses.pop(0)

    mock_client.chat.completions.create.side_effect = mock_create

    agent = AntiHallucinator(mock_client, strictness=0.5, tools=[mock_tool])
    result = agent.generate(model="dummy", prompt="Capital?")

    mock_tool.search.assert_not_called()
    assert result.verification_log[0]["evidence_used"] is False


def test_tool_failure_does_not_crash() -> None:
    mock_client = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "BrokenSearch"
    mock_tool.search.side_effect = RuntimeError("down")

    responses = [
        _mock_response("Python by Guido."),
        _mock_response("1. Python by Guido."),
        _mock_response("VERIFIED: True."),
    ]

    def mock_create(*args, **kwargs):
        return responses.pop(0)

    mock_client.chat.completions.create.side_effect = mock_create

    agent = AntiHallucinator(mock_client, strictness=1.0, tools=[mock_tool])
    result = agent.generate(model="dummy", prompt="Who?")

    assert result.content == "Python by Guido."
    assert result.verification_log[0]["evidence_used"] is False


# ------------------------------------------------------------------
# Async generate
# ------------------------------------------------------------------

def test_generate_async_basic() -> None:
    """generate_async should produce the same result as sync generate."""
    mock_client = MagicMock()

    responses = [
        _mock_response("Water boils at 100C."),
        _mock_response("1. Water boils at 100 degrees Celsius."),
        _mock_response("VERIFIED: True."),
    ]

    def mock_create(*args, **kwargs):
        return responses.pop(0)

    mock_client.chat.completions.create.side_effect = mock_create

    agent = AntiHallucinator(mock_client, strictness=1.0)
    result = asyncio.run(agent.generate_async(model="dummy", prompt="Water?"))

    assert result.content == "Water boils at 100C."
    assert len(result.hallucinations_caught) == 0
    assert result.token_usage.total_tokens > 0
    assert result.elapsed_seconds > 0


def test_generate_async_parallel_verification() -> None:
    """Async should verify multiple claims in parallel."""
    mock_client = MagicMock()

    responses = [
        _mock_response("Paris is capital. Water boils at 100C."),
        _mock_response("1. Paris is capital.\n2. Water boils at 100C."),
        _mock_response("VERIFIED: True."),  # For claim 1
        _mock_response("VERIFIED: True."),  # For claim 2
    ]

    def mock_create(*args, **kwargs):
        return responses.pop(0)

    mock_client.chat.completions.create.side_effect = mock_create

    agent = AntiHallucinator(mock_client, strictness=1.0)
    result = asyncio.run(agent.generate_async(model="dummy", prompt="Facts?"))

    # 4 calls total: draft + extract + 2 critiques (in parallel)
    assert mock_client.chat.completions.create.call_count == 4
    assert len(result.verification_log) == 2
    assert all(r["is_valid"] for r in result.verification_log)
