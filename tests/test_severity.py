"""Tests for flagged-claim severity classification."""

from __future__ import annotations

import json

import pytest

from self_correct.cli import _build_parser
from self_correct.core import (
    DEFAULT_SEVERITY_RULES,
    VALID_SEVERITIES,
    AntiHallucinationResponse,
    VerificationPolicy,
    classify_severity,
)


def test_default_rules_classify_explicit_falsehoods_as_critical() -> None:
    assert classify_severity("VERIFIED: False. The claim is incorrect.") == "critical"
    assert classify_severity("This contradicts recorded history") == "critical"
    assert classify_severity("The date was fabricated by the model") == "critical"


def test_default_rules_classify_uncertainty_as_major() -> None:
    assert classify_severity("Claim is unverifiable with available sources") == "major"
    assert classify_severity("Cannot be verified from public records") == "major"
    assert classify_severity("The figure is outdated") == "major"


def test_unmatched_critiques_fall_back_to_minor() -> None:
    assert classify_severity("Phrasing seems imprecise.") == "minor"
    assert classify_severity("") == "minor"


def test_matching_is_case_insensitive() -> None:
    assert classify_severity("MISLEADING comparison") == "critical"


def test_custom_rules_override_the_defaults() -> None:
    rules = [(r"severe", "critical")]
    assert classify_severity("severe problem", rules) == "critical"
    assert classify_severity("incorrect and false", rules) == "minor"


def test_unknown_severity_in_rules_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown severity"):
        classify_severity("text", [(r"x", "catastrophic")])


def test_valid_severities_are_the_documented_labels() -> None:
    assert VALID_SEVERITIES == ("critical", "major", "minor")
    for _, severity in DEFAULT_SEVERITY_RULES:
        assert severity in VALID_SEVERITIES


def test_severity_summary_counts_flagged_claims_only() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[
            {"claim": "a", "is_valid": False, "critique": "False and misleading"},
            {"claim": "b", "is_valid": False, "critique": "unverifiable claim"},
            {"claim": "c", "is_valid": False, "critique": "imprecise wording"},
            {"claim": "d", "is_valid": True, "critique": "False"},
            {"phase": "extraction", "warning": "none"},
        ],
    )
    assert response.severity_summary() == {
        "critical": 1,
        "major": 1,
        "minor": 1,
    }


def test_severity_summary_accepts_custom_rules() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[{"claim": "a", "is_valid": False, "critique": "wrong"}],
    )
    custom = [(r"wrong", "major")]
    assert response.severity_summary(custom)["major"] == 1
    assert response.severity_summary()["minor"] == 1


def test_empty_response_has_a_zeroed_summary() -> None:
    response = AntiHallucinationResponse(content="text")
    assert response.severity_summary() == {"critical": 0, "major": 0, "minor": 0}


def test_severity_summary_is_serialized() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[
            {"claim": "a", "is_valid": False, "critique": "fabricated data"},
        ],
    )
    payload = json.loads(response.to_json())
    assert payload["severity_summary"]["critical"] == 1
    assert payload["severity_summary"]["major"] == 0


def test_markdown_labels_each_flagged_claim_with_its_severity() -> None:
    response = AntiHallucinationResponse(
        content="text",
        hallucinations_caught=["Claim 'x' flagged: incorrect date"],
    )
    markdown = response.to_markdown()
    assert "1. [critical] Claim 'x' flagged: incorrect date" in markdown


def test_policy_gate_counts_critical_claims() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[
            {"claim": "a", "is_valid": False, "critique": "false statement"},
            {"claim": "b", "is_valid": False, "critique": "unverifiable claim"},
        ],
    )
    decision = response.evaluate(VerificationPolicy(min_verified_ratio=0.0, max_critical_claims=0))
    assert decision.passed is False
    assert decision.reasons == ["critical claims 1 exceed 0"]

    tolerant = response.evaluate(
        VerificationPolicy(min_verified_ratio=0.0, max_critical_claims=1, max_flagged_claims=2)
    )
    assert tolerant.passed is True


def test_policy_rejects_negative_critical_allowance() -> None:
    with pytest.raises(ValueError, match="max_critical_claims"):
        VerificationPolicy(max_critical_claims=-1)


def test_policy_loads_max_critical_claims_from_json(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text('{"max_critical_claims": 2}', encoding="utf-8")
    policy = VerificationPolicy.from_json(path)
    assert policy.max_critical_claims == 2


def test_policy_rejects_unknown_fields_still() -> None:
    with pytest.raises(ValueError, match="unknown verification policy fields"):
        VerificationPolicy.from_dict({"max_critical_count": 1})


def test_cli_policy_flag_remains_available() -> None:
    args = _build_parser().parse_args(["verify", "--prompt", "p"])
    assert hasattr(args, "policy")
