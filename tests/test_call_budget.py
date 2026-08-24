"""Tests for the per-run LLM call budget."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from self_correct.cli import _build_parser
from self_correct.core import AntiHallucinator, AntiHallucinationResponse


def _mock_response(content, prompt_tokens=10, completion_tokens=20):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def _agent_with_script(mock_client, **kwargs):
    return AntiHallucinator(mock_client, strictness=1.0, **kwargs)


def test_max_llm_calls_must_be_a_positive_integer() -> None:
    with pytest.raises(ValueError, match="max_llm_calls"):
        AntiHallucinator(MagicMock(), max_llm_calls=0)
    with pytest.raises(ValueError, match="max_llm_calls"):
        AntiHallucinator(MagicMock(), max_llm_calls=-2)
    with pytest.raises(ValueError, match="max_llm_calls"):
        AntiHallucinator(MagicMock(), max_llm_calls=True)


def test_unbudgeted_run_is_unaffected() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("Draft."),
        _mock_response("1. One claim."),
        _mock_response("VERIFIED: True."),
    ]
    agent = _agent_with_script(client)
    result = agent.generate(model="dummy", prompt="p")
    assert [e.get("is_valid") for e in result.verification_log] == [True]
    assert result.budget_report()["exhausted"] is False


def test_budget_report_is_empty_without_skip_entries() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[{"claim": "a", "is_valid": True}],
    )
    assert response.budget_report() == {
        "exhausted": False,
        "skipped_claims": [],
        "skipped_phases": [],
    }


def test_budget_skips_remaining_claims_in_sync_pipeline() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("Claim one. Claim two."),          # draft
        _mock_response("1. Claim one.\n2. Claim two."),   # extraction
        _mock_response("VERIFIED: True."),                # claim 1
        # claim 2 would need a fifth... fourth call: budget spent
    ]
    agent = _agent_with_script(client, max_llm_calls=3)
    result = agent.generate(model="dummy", prompt="p")

    assert client.chat.completions.create.call_count == 3
    report = result.budget_report()
    assert report["exhausted"] is True
    assert report["skipped_claims"] == ["Claim two."]
    assert report["skipped_phases"] == []
    skipped_entry = result.verification_log[-1]
    assert skipped_entry["skipped_by_budget"] is True
    assert "is_valid" not in skipped_entry
    assert result.claim_summary()["total_claims"] == 1


def test_budget_skips_correction_but_keeps_flagged_claims() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("Claim one."),                     # draft
        _mock_response("1. Claim one."),                  # extraction
        _mock_response("VERIFIED: False. Fabricated."),   # critique
    ]
    agent = _agent_with_script(client, max_llm_calls=3)
    result = agent.generate(model="dummy", prompt="p")

    assert len(result.hallucinations_caught) == 1
    assert result.content == "Claim one."
    report = result.budget_report()
    assert report["exhausted"] is True
    assert report["skipped_phases"] == ["correction"]
    assert report["skipped_claims"] == []


def test_exhaustion_before_extraction_halts_the_pipeline() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("Just a draft."),
    ]
    agent = _agent_with_script(client, max_llm_calls=1)
    result = agent.generate(model="dummy", prompt="p")

    assert client.chat.completions.create.call_count == 1
    assert result.content == "Just a draft."
    assert result.verification_log[0]["phase"] == "budget"
    assert result.claim_summary()["total_claims"] == 0
    assert result.budget_report()["exhausted"] is True


def test_cached_verifications_do_not_consume_the_budget() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("Paris is in France."),
        _mock_response("1. Paris is in France."),
        _mock_response("VERIFIED: True."),
    ]
    agent = _agent_with_script(client, max_llm_calls=3)
    agent.generate(model="dummy", prompt="p")

    client.chat.completions.create.reset_mock()
    client.chat.completions.create.side_effect = [
        _mock_response("Paris is in France."),
        _mock_response("1. Paris is in France."),
    ]
    result = agent.generate(model="dummy", prompt="p")
    assert client.chat.completions.create.call_count == 2
    assert result.verification_log[0].get("cached") is True
    assert result.budget_report()["exhausted"] is False


def test_budget_applies_to_async_verification() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("Claim one. Claim two."),
        _mock_response("1. Claim one.\n2. Claim two."),
        _mock_response("VERIFIED: True."),
    ]
    agent = _agent_with_script(client, max_llm_calls=3)
    result = asyncio.run(agent.generate_async(model="dummy", prompt="p"))

    assert client.chat.completions.create.call_count == 3
    report = result.budget_report()
    assert report["exhausted"] is True
    assert report["skipped_claims"] == ["Claim two."]


def test_async_correction_respects_the_budget() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("Claim one."),
        _mock_response("1. Claim one."),
        _mock_response("VERIFIED: False. Incorrect."),
    ]
    agent = _agent_with_script(client, max_llm_calls=3)
    result = asyncio.run(agent.generate_async(model="dummy", prompt="p"))

    assert result.hallucinations_caught
    assert result.content == "Claim one."
    assert result.budget_report()["skipped_phases"] == ["correction"]


def test_budget_is_serialized_in_to_dict() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[{"claim": "x", "skipped_by_budget": True}],
    )
    payload = json.loads(response.to_json())
    assert payload["budget"]["exhausted"] is True
    assert payload["budget"]["skipped_claims"] == ["x"]


def test_markdown_reports_an_exhausted_budget() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[
            {"claim": "a", "is_valid": True},
            {"claim": "b", "skipped_by_budget": True},
        ],
    )
    markdown = response.to_markdown()
    assert "**LLM call budget**: exhausted; 1 claim(s) left unverified" in markdown


def test_markdown_omits_budget_line_when_intact() -> None:
    response = AntiHallucinationResponse(content="clean")
    assert "LLM call budget" not in response.to_markdown()


def test_cli_accepts_max_calls_on_verify_resume_and_batch() -> None:
    parser = _build_parser()
    args = parser.parse_args(["verify", "--prompt", "p", "--max-calls", "12"])
    assert args.max_calls == 12
    args = parser.parse_args(["resume", "s.json", "--max-calls", "5"])
    assert args.max_calls == 5
    args = parser.parse_args(["batch", "--input", "in.jsonl", "--max-calls", "7"])
    assert args.max_calls == 7
