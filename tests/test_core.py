"""Tests for self_correct.core.AntiHallucinator."""

import asyncio
import threading
import time
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
from self_correct.core import (
    AntiHallucinator,
    AntiHallucinationResponse,
    TokenUsage,
    VerificationPolicy,
    _ClaimCache,
)


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


def test_cache_scopes_prevent_cross_configuration_reuse() -> None:
    cache = _ClaimCache(max_size=10)
    cache.put("The sky is blue", {"model": "small"}, scope="model=small")

    assert cache.get("the sky is blue", scope="model=small") == {"model": "small"}
    assert cache.get("the sky is blue", scope="model=large") is None


def test_unscoped_cache_keys_remain_backward_compatible() -> None:
    cache = _ClaimCache(max_size=10)
    cache.put("The sky is blue", {"is_valid": True})
    assert cache.get("the sky is blue") == {"is_valid": True}


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


def test_claim_cache_persists_between_agents(tmp_path) -> None:
    first = AntiHallucinator(MagicMock(), cache_size=2)
    first.cache.put("Paris is in France", {"is_valid": True, "critique": "ok"})
    snapshot = first.save_cache(tmp_path / "nested" / "claims.json")

    second = AntiHallucinator(MagicMock(), cache_size=2)
    assert second.load_cache(snapshot) == 1
    assert second.cache.get("paris is in france") == {
        "is_valid": True,
        "critique": "ok",
    }


def test_claim_cache_respects_destination_capacity(tmp_path) -> None:
    source = AntiHallucinator(MagicMock(), cache_size=3)
    for claim in ("A", "B", "C"):
        source.cache.put(claim, {"claim": claim})
    snapshot = source.save_cache(tmp_path / "claims.json")

    destination = AntiHallucinator(MagicMock(), cache_size=2)
    assert destination.load_cache(snapshot) == 3
    assert destination.cache_size == 2
    assert destination.cache.get("A") is None


def test_claim_cache_snapshot_can_be_updated_in_place(tmp_path) -> None:
    path = tmp_path / "claims.json"
    first = AntiHallucinator(MagicMock())
    first.cache.put("Claim A", {"is_valid": True})
    first.save_cache(path)

    second = AntiHallucinator(MagicMock())
    second.load_cache(path)
    second.cache.put("Claim B", {"is_valid": False})
    second.save_cache(path)

    restored = AntiHallucinator(MagicMock())
    assert restored.load_cache(path) == 2
    assert restored.cache.get("Claim A") is not None
    assert restored.cache.get("Claim B") is not None


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def test_strictness_clamped_high() -> None:
    agent = AntiHallucinator(MagicMock(), strictness=2.0)
    assert agent.strictness == 1.0


def test_strictness_clamped_low() -> None:
    agent = AntiHallucinator(MagicMock(), strictness=-1.0)
    assert agent.strictness == 0.0


def test_verification_policy_explains_failed_release_gate() -> None:
    response = AntiHallucinationResponse(
        content="Corrected",
        hallucinations_caught=["bad claim"],
        verification_log=[
            {"claim": "good", "is_valid": True},
            {"claim": "bad", "is_valid": False},
        ],
    )
    decision = response.evaluate(
        VerificationPolicy(min_verified_ratio=0.75, max_flagged_claims=0)
    )
    assert decision.passed is False
    assert decision.verified_ratio == 0.5
    assert decision.reasons == [
        "verified ratio 50.0% is below 75.0%",
        "flagged claims 1 exceed 0",
    ]


def test_verification_policy_can_require_at_least_one_claim() -> None:
    response = AntiHallucinationResponse(content="No factual claims")
    decision = response.evaluate(VerificationPolicy(require_claims=True))
    assert decision.to_dict()["passed"] is False
    assert decision.reasons == ["no factual claims were verified"]


def test_verification_policy_can_require_evidence_coverage() -> None:
    response = AntiHallucinationResponse(
        content="Answer",
        verification_log=[
            {"claim": "one", "is_valid": True, "evidence_used": True},
            {"claim": "two", "is_valid": True, "evidence_used": False},
        ],
    )
    decision = response.evaluate(VerificationPolicy(min_evidence_ratio=1.0))

    assert decision.passed is False
    assert decision.evidence_claims == 1
    assert decision.evidence_ratio == 0.5
    assert decision.reasons == ["evidence ratio 50.0% is below 100.0%"]


def test_verification_policy_can_require_an_absolute_verified_claim_count() -> None:
    response = AntiHallucinationResponse(
        content="Answer",
        verification_log=[{"claim": "one", "is_valid": True}],
    )

    decision = response.evaluate(VerificationPolicy(min_verified_claims=2))

    assert decision.passed is False
    assert decision.reasons == ["verified claims 1 are below 2"]


def test_verification_policy_can_limit_hallucination_density() -> None:
    response = AntiHallucinationResponse(
        content="word " * 10,
        hallucinations_caught=["flagged"],
    )

    decision = response.evaluate(
        VerificationPolicy(max_flagged_claims=1, max_hallucination_density=5.0)
    )

    assert decision.passed is False
    assert decision.reasons == [
        "hallucination density 10.00 exceeds 5.00 per 100 words"
    ]


def test_verification_policy_loads_from_json(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        '{"min_verified_ratio": 0.8, "max_flagged_claims": 1, "require_claims": true, "min_evidence_ratio": 0.5}',
        encoding="utf-8",
    )
    policy = VerificationPolicy.from_json(path)
    assert policy.min_verified_ratio == 0.8
    assert policy.max_flagged_claims == 1
    assert policy.require_claims is True
    assert policy.min_evidence_ratio == 0.5


def test_verification_policy_rejects_invalid_evidence_ratio() -> None:
    with pytest.raises(ValueError, match="min_evidence_ratio"):
        VerificationPolicy(min_evidence_ratio=1.1)


def test_verification_policy_rejects_negative_verified_claim_minimum() -> None:
    with pytest.raises(ValueError, match="min_verified_claims"):
        VerificationPolicy(min_verified_claims=-1)


def test_verification_policy_rejects_negative_hallucination_density_limit() -> None:
    with pytest.raises(ValueError, match="max_hallucination_density"):
        VerificationPolicy(max_hallucination_density=-0.1)


def test_verification_policy_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown verification policy fields"):
        VerificationPolicy.from_dict({"minimum_score": 0.9})


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


def test_call_llm_retries_transient_provider_failure(monkeypatch) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        ConnectionError("temporary"),
        _mock_response("recovered"),
    ]
    sleeps = []
    monkeypatch.setattr("self_correct.core.time.sleep", sleeps.append)
    agent = AntiHallucinator(mock_client, max_retries=1, retry_backoff=0.25)

    assert agent._call_llm("dummy", "sys", "usr") == "recovered"
    assert mock_client.chat.completions.create.call_count == 2
    assert sleeps == [0.25]


def test_call_llm_retry_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        AntiHallucinator(MagicMock(), max_retries=-1)
    with pytest.raises(ValueError, match="retry_backoff"):
        AntiHallucinator(MagicMock(), retry_backoff=-0.1)


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


def test_cache_scope_changes_with_model_and_tools() -> None:
    tool = MagicMock()
    tool.name = "Search"
    first = AntiHallucinator(MagicMock(), tools=[tool])
    second = AntiHallucinator(MagicMock())
    prompt = first._build_critique_prompt()

    assert first._cache_scope("model-a", prompt, True) != first._cache_scope(
        "model-b", prompt, True
    )
    assert first._cache_scope("model-a", prompt, True) != second._cache_scope(
        "model-a", prompt, False
    )


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
    assert result.verification_log[0]["evidence_sources"] == [
        {"title": "Page", "url": "https://x.com", "tool": "MockSearch"}
    ]


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


def test_generate_async_honors_max_concurrency() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("Claim one. Claim two. Claim three."),
        _mock_response("1. Claim one.\n2. Claim two.\n3. Claim three."),
    ]
    agent = AntiHallucinator(mock_client, strictness=1.0)
    active = 0
    peak = 0
    lock = threading.Lock()

    def verify(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {
            "claim": args[0],
            "is_valid": True,
            "critique": "VERIFIED: True.",
            "evidence_used": False,
            "cached": False,
        }

    agent._verify_single_claim = verify
    result = asyncio.run(
        agent.generate_async(model="dummy", prompt="Facts?", max_concurrency=2)
    )

    assert len(result.verification_log) == 3
    assert peak == 2


def test_generate_async_rejects_invalid_concurrency() -> None:
    agent = AntiHallucinator(MagicMock())
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(agent.generate_async("dummy", "prompt", max_concurrency=0))


def test_generate_many_preserves_prompt_order_and_options() -> None:
    agent = AntiHallucinator(MagicMock())
    agent.generate = MagicMock(
        side_effect=[
            AntiHallucinationResponse(content="first"),
            AntiHallucinationResponse(content="second"),
        ]
    )

    results = agent.generate_many("dummy", ["Prompt one", "Prompt two"], max_tokens=64)

    assert [result.content for result in results] == ["first", "second"]
    assert agent.generate.call_args_list[0].kwargs == {
        "model": "dummy", "prompt": "Prompt one", "max_tokens": 64
    }


def test_generate_many_rejects_empty_prompt_entries() -> None:
    agent = AntiHallucinator(MagicMock())
    with pytest.raises(ValueError, match="non-empty"):
        agent.generate_many("dummy", ["valid", "  "])


class TestClaimCacheTTL:
    """The claim cache may expire entries after a configurable interval."""

    def test_entry_survives_within_ttl(self):
        from self_correct.core import _ClaimCache

        cache = _ClaimCache(max_size=8, ttl=30.0)
        cache.put("the sky is blue", {"valid": True})
        assert cache.get("the sky is blue") == {"valid": True}

    def test_entry_expires_after_ttl(self):
        import time

        from self_correct.core import _ClaimCache

        cache = _ClaimCache(max_size=8, ttl=0.05)
        cache.put("the sky is blue", {"valid": True})
        time.sleep(0.1)
        assert cache.get("the sky is blue") is None

    def test_no_ttl_means_no_expiry(self):
        from self_correct.core import _ClaimCache

        cache = _ClaimCache(max_size=8, ttl=None)
        cache.put("the sky is blue", {"valid": True})
        assert cache.ttl is None
        assert cache.get("the sky is blue") == {"valid": True}

    def test_lru_eviction_still_applies(self):
        from self_correct.core import _ClaimCache

        cache = _ClaimCache(max_size=2)
        for claim in ("a", "b", "c"):
            cache.put(claim, {"claim": claim})
        assert cache.get("a") is None
        assert cache.get("c") == {"claim": "c"}

    def test_stats_track_hits_misses_and_expirations(self):
        import time

        from self_correct.core import _ClaimCache

        cache = _ClaimCache(max_size=8, ttl=0.05)
        cache.put("x", {"valid": True})
        cache.get("x")            # hit
        cache.get("absent")       # miss
        time.sleep(0.1)
        cache.get("x")            # expired -> miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["expirations"] == 1
        assert stats["ttl_seconds"] == 0.05
        assert 0.0 < stats["hit_rate"] < 1.0


class TestModelPricing:
    def test_exact_match(self):
        from self_correct.core import model_pricing

        assert model_pricing("gpt-4o-mini") == (0.15, 0.60)

    def test_dated_suffix_resolves_to_base_model(self):
        from self_correct.core import model_pricing

        assert model_pricing("gpt-4o-mini-2024-07-18") == model_pricing("gpt-4o-mini")

    def test_longest_prefix_wins(self):
        """gpt-4-turbo-preview must not be priced as gpt-4."""
        from self_correct.core import model_pricing

        assert model_pricing("gpt-4-turbo-preview") == model_pricing("gpt-4-turbo")
        assert model_pricing("gpt-4-turbo-preview") != model_pricing("gpt-4")

    def test_unknown_model_returns_none(self):
        from self_correct.core import model_pricing

        assert model_pricing("llama-3-70b") is None

    def test_cost_uses_the_models_own_rates(self):
        from self_correct.core import TokenUsage

        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert usage.estimate_cost_for_model("gpt-4o-mini") == 0.15 + 0.60

    def test_cost_is_none_for_unpriced_model(self):
        """Better to report unknown than to quote another model's rates."""
        from self_correct.core import TokenUsage

        usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000)
        assert usage.estimate_cost_for_model("llama-3-70b") is None

# ------------------------------------------------------------------
# Hallucination density scoring
# ------------------------------------------------------------------

def test_density_counts_flagged_claims_per_words() -> None:
    response = AntiHallucinationResponse(
        content="one two three four five six seven eight nine ten " * 10,
        hallucinations_caught=["a", "b"],
    )
    assert response.hallucination_density() == 2.0

def test_density_scales_with_chunk_size() -> None:
    response = AntiHallucinationResponse(
        content="word " * 100,
        hallucinations_caught=["x"],
    )
    assert response.hallucination_density(per_words=50) == 0.5

def test_density_zero_for_empty_response() -> None:
    response = AntiHallucinationResponse(content="")
    assert response.hallucination_density() == 0.0

def test_density_rejects_non_positive_chunk() -> None:
    response = AntiHallucinationResponse(content="some words")
    with pytest.raises(ValueError, match="per_words"):
        response.hallucination_density(per_words=0)

def test_density_is_serialized() -> None:
    import json as _json

    response = AntiHallucinationResponse(
        content="hello world",
        hallucinations_caught=["wrong claim"],
    )
    payload = _json.loads(response.to_json())
    assert payload["hallucination_density"] == 50.0

def test_density_is_reported_in_markdown() -> None:
    response = AntiHallucinationResponse(content="clean answer")
    markdown = response.to_markdown()
    assert "Hallucination density" in markdown

# ------------------------------------------------------------------
# Claim verdict summary
# ------------------------------------------------------------------

def test_claim_summary_counts_verdicts() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[
            {"claim": "a", "is_valid": True, "evidence_used": True, "critique": ""},
            {"claim": "b", "is_valid": False, "evidence_used": True, "critique": "no"},
            {"claim": "c", "is_valid": True, "evidence_used": False, "critique": ""},
        ],
    )
    assert response.claim_summary() == {
        "total_claims": 3,
        "verified_claims": 2,
        "flagged_claims": 1,
        "evidence_claims": 2,
    }

def test_claim_summary_ignores_phase_entries() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[
            {"phase": "extraction", "warning": "No claims extracted."},
            {"claim": "a", "is_valid": True, "evidence_used": False, "critique": ""},
        ],
    )
    assert response.claim_summary()["total_claims"] == 1

def test_claim_summary_empty_log() -> None:
    response = AntiHallucinationResponse(content="text")
    assert response.claim_summary() == {
        "total_claims": 0,
        "verified_claims": 0,
        "flagged_claims": 0,
        "evidence_claims": 0,
    }

def test_claim_summary_is_serialized() -> None:
    import json as _json

    response = AntiHallucinationResponse(
        content="text",
        verification_log=[
            {"claim": "a", "is_valid": True, "evidence_used": False, "critique": ""},
        ],
    )
    payload = _json.loads(response.to_json())
    assert payload["claim_summary"]["verified_claims"] == 1


def test_evidence_summary_deduplicates_source_urls_and_lists_tools() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[
            {"evidence_sources": [{"url": "https://a", "tool": "Search"}]},
            {"evidence_sources": [{"url": "https://a", "tool": "Search"}, {"url": "https://b", "tool": "Archive"}]},
        ],
    )
    assert response.evidence_summary() == {
        "source_count": 3,
        "unique_url_count": 2,
        "tools": ["Archive", "Search"],
    }
    assert response.to_dict()["evidence_summary"]["source_count"] == 3
