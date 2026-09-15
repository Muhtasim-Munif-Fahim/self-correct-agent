import pytest

from self_correct.core import claim_risk_summary


def test_summary_ranks_claims_and_counts_high_risk_items():
    result = claim_risk_summary(["It is useful.", "Revenue was $5m in 2023."], high_risk=3)
    assert result["count"] == 2
    assert result["high_risk_count"] == 1
    assert result["claims"][0]["claim"] == "Revenue was $5m in 2023."
    assert result["mean_risk"] > 0


def test_summary_handles_empty_input():
    assert claim_risk_summary([]) == {
        "count": 0,
        "high_risk_threshold": 3.0,
        "high_risk_count": 0,
        "mean_risk": 0.0,
        "claims": [],
    }


@pytest.mark.parametrize("threshold", [True, "3"])
def test_summary_validates_threshold(threshold):
    with pytest.raises(ValueError, match="high_risk"):
        claim_risk_summary([], high_risk=threshold)
