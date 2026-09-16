"""Tests for JSON mode / function-calling structured verification output."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from self_correct import AntiHallucinator, StructuredClaim, StructuredVerificationResult
from self_correct.cli import _build_parser
from self_correct.core import AntiHallucinationResponse
from self_correct.structured import (
    CLAIM_VERDICT_SCHEMA,
    EXTRACT_CLAIMS_SCHEMA,
    normalize_structured_mode,
    parse_claim_verdict,
    parse_extracted_claims,
    parse_json_text,
    payload_from_response,
    request_kwargs,
)


def _mock_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    resp.choices[0].message.function_call = None
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def _json_response(payload: dict, prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    return _mock_response(json.dumps(payload), prompt_tokens, completion_tokens)


def _function_response(payload: dict, prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = None
    call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps(payload)))
    resp.choices[0].message.tool_calls = [call]
    resp.choices[0].message.function_call = None
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


# ------------------------------------------------------------------
# Helpers / schema
# ------------------------------------------------------------------

def test_normalize_structured_mode_aliases() -> None:
    assert normalize_structured_mode(False) is None
    assert normalize_structured_mode(None) is None
    assert normalize_structured_mode(True) == "json"
    assert normalize_structured_mode("JSON") == "json"
    assert normalize_structured_mode("function_call") == "function"
    with pytest.raises(ValueError, match="unknown structured_output"):
        normalize_structured_mode("xml")
    with pytest.raises(ValueError, match="must be False"):
        normalize_structured_mode(1)


def test_request_kwargs_json_and_function() -> None:
    assert request_kwargs("json", "extract") == {"response_format": {"type": "json_object"}}
    fn = request_kwargs("function", "verify")
    assert fn["tool_choice"]["function"]["name"] == "verify_claim"
    assert fn["tools"][0]["function"]["parameters"] == CLAIM_VERDICT_SCHEMA
    extract = request_kwargs("function", "extract")
    assert extract["tools"][0]["function"]["parameters"] == EXTRACT_CLAIMS_SCHEMA


def test_parse_json_text_accepts_fenced_blocks() -> None:
    text = "```json\n{\"claims\": [\"A\"]}\n```"
    assert parse_json_text(text) == {"claims": ["A"]}
    assert parse_json_text("not json") is None
    assert parse_extracted_claims({"claims": ["  A  ", "", 3]}) == ["A", "3"]
    assert parse_extracted_claims({"nope": []}) is None
    assert parse_claim_verdict({"is_valid": "true", "critique": "ok"}) == {
        "is_valid": True,
        "critique": "ok",
    }
    assert parse_claim_verdict({"verified": False, "reason": "no"}) == {
        "is_valid": False,
        "critique": "no",
    }


def test_payload_from_response_reads_tool_calls() -> None:
    response = _function_response({"is_valid": False, "critique": "wrong"})
    assert payload_from_response(response) == {"is_valid": False, "critique": "wrong"}


def test_constructor_rejects_invalid_structured_mode() -> None:
    with pytest.raises(ValueError, match="structured_output"):
        AntiHallucinator(MagicMock(), structured_output="yaml")


# ------------------------------------------------------------------
# Pipeline: JSON mode
# ------------------------------------------------------------------

def test_json_mode_extracts_claims_and_verdicts() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("Paris is the capital of France."),
        _json_response({"claims": ["Paris is the capital of France."]}),
        _json_response({"is_valid": True, "critique": "Well established."}),
    ]
    agent = AntiHallucinator(mock_client, structured_output=True)
    result = agent.generate(model="dummy", prompt="Capital?")

    assert result.content == "Paris is the capital of France."
    assert result.verification_log[0]["is_valid"] is True
    assert result.verification_log[0]["structured"] is True
    assert result.verification_log[0]["critique"] == "Well established."
    assert agent.structured_output == "json"

    calls = mock_client.chat.completions.create.call_args_list
    assert "response_format" not in calls[0].kwargs
    assert calls[1].kwargs["response_format"] == {"type": "json_object"}
    assert calls[2].kwargs["response_format"] == {"type": "json_object"}
    assert "json" in calls[1].kwargs["messages"][0]["content"].lower()


def test_json_mode_flags_invalid_claims_and_corrects() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("The moon is cheese."),
        _json_response({"claims": ["The moon is cheese."]}),
        _json_response({"is_valid": False, "critique": "The moon is rock."}),
        _mock_response("The moon is rock."),
    ]
    agent = AntiHallucinator(mock_client, structured_output="json")
    typed = agent.generate_structured(model="dummy", prompt="Moon?")

    assert typed.content == "The moon is rock."
    assert len(typed.claims) == 1
    assert typed.claims[0].is_valid is False
    assert typed.claims[0].structured is True
    assert typed.hallucinations_caught
    calls = mock_client.chat.completions.create.call_args_list
    assert "response_format" not in calls[3].kwargs


def test_json_mode_falls_back_to_text_parsers() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("Water boils at 100C."),
        _mock_response("1. Water boils at 100C."),
        _mock_response("VERIFIED: True."),
    ]
    agent = AntiHallucinator(mock_client, structured_output="json")
    result = agent.generate(model="dummy", prompt="Water?")

    assert result.verification_log[0]["is_valid"] is True
    assert result.verification_log[0]["structured"] is False


def test_json_mode_retries_plain_text_when_client_rejects_format() -> None:
    mock_client = MagicMock()

    def create(**kwargs):
        if "response_format" in kwargs:
            raise TypeError("unexpected keyword argument 'response_format'")
        user = kwargs["messages"][1]["content"]
        system = kwargs["messages"][0]["content"]
        if "Text to analyze" in user:
            return _mock_response("1. Paris is the capital of France.")
        if "Claim:" in user or "skepticism" in system.lower() or "factual errors" in system.lower():
            return _mock_response("VERIFIED: True.")
        return _mock_response("Paris is the capital of France.")

    mock_client.chat.completions.create.side_effect = create
    agent = AntiHallucinator(mock_client, structured_output=True)
    result = agent.generate(model="dummy", prompt="Capital?")

    assert result.content == "Paris is the capital of France."
    assert result.verification_log[0]["is_valid"] is True
    assert any(
        "response_format" in (call.kwargs or {})
        for call in mock_client.chat.completions.create.call_args_list
    )


def test_default_path_does_not_send_response_format() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("Draft."),
        _mock_response("1. Draft."),
        _mock_response("VERIFIED: True."),
    ]
    AntiHallucinator(mock_client).generate(model="dummy", prompt="Hi")
    for call in mock_client.chat.completions.create.call_args_list:
        assert "response_format" not in call.kwargs
        assert "tools" not in call.kwargs


# ------------------------------------------------------------------
# Pipeline: function calling
# ------------------------------------------------------------------

def test_function_calling_extracts_and_verifies() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("Paris is the capital of France."),
        _function_response({"claims": ["Paris is the capital of France."]}),
        _function_response({"is_valid": True, "critique": "Confirmed."}),
    ]
    agent = AntiHallucinator(mock_client, structured_output="function")
    result = agent.generate(model="dummy", prompt="Capital?")

    assert result.verification_log[0]["is_valid"] is True
    assert result.verification_log[0]["structured"] is True
    assert result.verification_log[0]["critique"] == "Confirmed."

    calls = mock_client.chat.completions.create.call_args_list
    assert "tools" not in calls[0].kwargs
    assert calls[1].kwargs["tools"][0]["function"]["name"] == "extract_claims"
    assert calls[1].kwargs["tool_choice"]["function"]["name"] == "extract_claims"
    assert calls[2].kwargs["tools"][0]["function"]["name"] == "verify_claim"


def test_function_calling_async_pipeline() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("Water boils at 100C."),
        _function_response({"claims": ["Water boils at 100C."]}),
        _function_response({"is_valid": True, "critique": "ok"}),
    ]
    agent = AntiHallucinator(mock_client, structured_output="function")
    typed = asyncio.run(agent.generate_structured_async(model="dummy", prompt="Water?"))
    assert typed.claims[0].is_valid is True
    assert typed.claims[0].structured is True


def test_cache_scope_includes_structured_mode() -> None:
    first = AntiHallucinator(MagicMock(), structured_output="json")
    second = AntiHallucinator(MagicMock())
    prompt = first._build_critique_prompt()
    assert first._cache_scope("m", prompt, False) != second._cache_scope("m", prompt, False)


# ------------------------------------------------------------------
# Typed result API
# ------------------------------------------------------------------

def test_to_structured_validates_response_schema() -> None:
    response = AntiHallucinationResponse(
        content="Corrected",
        hallucinations_caught=["bad claim"],
        verification_log=[
            {"claim": "good", "is_valid": True, "critique": "ok", "structured": True},
            {"claim": "bad", "is_valid": False, "critique": "no"},
            {"phase": "budget", "skipped_by_budget": True},
        ],
    )
    typed = response.to_structured()
    assert isinstance(typed, StructuredVerificationResult)
    assert [claim.claim for claim in typed.claims] == ["good", "bad"]
    assert typed.claim_summary["flagged_claims"] == 1
    roundtrip = StructuredVerificationResult.from_dict(typed.to_dict())
    assert roundtrip.content == "Corrected"
    assert roundtrip.claims[0].structured is True

    model = typed.to_pydantic()
    assert model.content == "Corrected"
    assert model.claims[0].is_valid is True
    dumped = model.model_dump() if hasattr(model, "model_dump") else model.dict()
    assert dumped["hallucinations_caught"] == ["bad claim"]


def test_structured_claim_from_dict_coerces_values() -> None:
    claim = StructuredClaim.from_dict(
        {
            "claim": "X",
            "is_valid": "false",
            "evidence_used": 1,
            "evidence_sources": [{"title": "T", "url": "u", "tool": "s"}],
        }
    )
    assert claim.is_valid is False
    assert claim.evidence_used is True
    assert claim.to_dict()["evidence_sources"][0]["url"] == "u"


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def test_cli_accepts_structured_output_flags() -> None:
    parser = _build_parser()
    json_args = parser.parse_args(["verify", "--prompt", "p", "--structured-output"])
    assert json_args.structured_output == "json"
    fn_args = parser.parse_args(
        ["verify", "--prompt", "p", "--structured-output", "function"]
    )
    assert fn_args.structured_output == "function"
    batch_args = parser.parse_args(
        ["batch", "--input", "in.jsonl", "--structured-output", "json"]
    )
    assert batch_args.structured_output == "json"
    resume_args = parser.parse_args(["resume", "s.json", "--structured-output"])
    assert resume_args.structured_output == "json"
    off_args = parser.parse_args(["verify", "--prompt", "p"])
    assert off_args.structured_output is None


def test_verify_cli_passes_structured_output_to_agent(tmp_path, monkeypatch) -> None:
    from self_correct import cli

    captured: dict = {}

    class FakeHallucinator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, **kwargs):
            return AntiHallucinationResponse(content="ok")

    monkeypatch.setenv("SELF_CORRECT_HISTORY", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(cli, "_build_client", lambda args: object())
    monkeypatch.setattr(cli, "AntiHallucinator", FakeHallucinator)
    assert (
        cli.main(
            [
                "verify",
                "--prompt",
                "Explain transformers.",
                "--structured-output",
                "function",
                "--output-format",
                "json",
            ]
        )
        == 0
    )
    assert captured["structured_output"] == "function"
