import pytest

from self_correct.core import explain_claim_risk


def test_explanation_exposes_the_signals_behind_a_score():
    result = explain_claim_risk("Revenue was $5 million in 2023.")
    assert {item["signal"] for item in result["signals"]} >= {"number", "year", "money"}
    assert result["risk_score"] > 0


def test_explanation_of_vague_claim_is_empty():
    assert explain_claim_risk("It is helpful.")["signals"] == []


def test_explanation_requires_text():
    with pytest.raises(ValueError, match="claim"):
        explain_claim_risk(None)
