"""Tests for risk-based claim prioritization under an LLM call budget."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from self_correct.core import AntiHallucinator, claim_risk_score, prioritize_claims


def _mock_response(content, prompt_tokens=10, completion_tokens=20):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


class TestClaimRiskScore:
    def test_a_vague_claim_scores_zero(self) -> None:
        assert claim_risk_score("The sky is blue.") == 0.0

    def test_numbers_outrank_bare_prose(self) -> None:
        assert claim_risk_score("Revenue was 42 million.") > claim_risk_score("Revenue grew.")

    @pytest.mark.parametrize(
        "claim",
        [
            "Growth was 42%.",
            "It shipped in 2019.",
            "Released on Mar 4 that year.",
            "The deal was worth $5 billion.",
            "According to the filing, it doubled.",
            "It was the first of its kind.",
        ],
    )
    def test_checkable_detail_scores_above_zero(self, claim: str) -> None:
        assert claim_risk_score(claim) > 0.0

    def test_each_pattern_counts_once_however_often_it_matches(self) -> None:
        one = claim_risk_score("There were 5 items.")
        many = claim_risk_score("There were 5, 6, 7, 8, 9 and 10 items.")
        assert one == many

    def test_stacked_signals_outrank_a_single_one(self) -> None:
        rich = claim_risk_score("According to Gartner, revenue grew 42% in 2023.")
        plain = claim_risk_score("Revenue grew 42.")
        assert rich > plain

    def test_a_sentence_initial_capital_is_not_a_proper_noun(self) -> None:
        # Otherwise every claim would score for its own first word.
        assert claim_risk_score("Paris is nice.") == 0.0

    def test_a_mid_sentence_proper_noun_does_score(self) -> None:
        assert claim_risk_score("The capital is Paris.") > 0.0

    def test_non_string_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="claim must be a string"):
            claim_risk_score(None)


class TestPrioritizeClaims:
    def test_risky_claims_come_first(self) -> None:
        claims = ["It is nice.", "Revenue grew 42% in 2023."]
        assert prioritize_claims(claims)[0] == "Revenue grew 42% in 2023."

    def test_ties_keep_their_original_order(self) -> None:
        claims = ["Alpha is fine.", "Beta is fine.", "Gamma is fine."]
        assert prioritize_claims(claims) == claims

    def test_the_input_list_is_not_mutated(self) -> None:
        claims = ["It is nice.", "Revenue grew 42%."]
        original = list(claims)
        prioritize_claims(claims)
        assert claims == original

    def test_empty_input(self) -> None:
        assert prioritize_claims([]) == []


class TestBudgetedVerificationOrder:
    CLAIMS = "1. The product is popular.\n2. Revenue grew 42% in 2023."

    def _client(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response("Draft."),
            _mock_response(self.CLAIMS),
            _mock_response("VERIFIED: True."),
            _mock_response("VERIFIED: True."),
            _mock_response("Corrected."),
        ]
        return client

    def test_the_budget_is_spent_on_the_checkable_claim(self) -> None:
        # Budget of 3 covers draft + extract + exactly one verification.
        agent = AntiHallucinator(self._client(), strictness=1.0, max_llm_calls=3)
        result = agent.generate(model="dummy", prompt="p")
        skipped = [e["claim"] for e in result.verification_log if e.get("skipped_by_budget")]
        verified = [e["claim"] for e in result.verification_log if "is_valid" in e]
        assert verified == ["Revenue grew 42% in 2023."]
        assert skipped == ["The product is popular."]

    def test_opting_out_falls_back_to_extraction_order(self) -> None:
        agent = AntiHallucinator(
            self._client(), strictness=1.0, max_llm_calls=3, prioritize_claims_by_risk=False
        )
        result = agent.generate(model="dummy", prompt="p")
        verified = [e["claim"] for e in result.verification_log if "is_valid" in e]
        assert verified == ["The product is popular."]

    def test_the_log_stays_in_extraction_order(self) -> None:
        agent = AntiHallucinator(self._client(), strictness=1.0, max_llm_calls=3)
        result = agent.generate(model="dummy", prompt="p")
        assert [e["claim"] for e in result.verification_log] == [
            "The product is popular.",
            "Revenue grew 42% in 2023.",
        ]

    def test_an_unbudgeted_run_verifies_everything_in_order(self) -> None:
        agent = AntiHallucinator(self._client(), strictness=1.0)
        result = agent.generate(model="dummy", prompt="p")
        assert [e["claim"] for e in result.verification_log] == [
            "The product is popular.",
            "Revenue grew 42% in 2023.",
        ]
        assert all("is_valid" in e for e in result.verification_log)

    def test_a_flagged_claim_still_triggers_correction(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response("Draft."),
            _mock_response(self.CLAIMS),
            # The prioritized (risky) claim is verified first and fails.
            _mock_response("VERIFIED: False. The figure is wrong."),
            _mock_response("VERIFIED: True."),
            _mock_response("Corrected."),
        ]
        # draft + extract + both verifications + correction.
        agent = AntiHallucinator(client, strictness=1.0, max_llm_calls=5)
        result = agent.generate(model="dummy", prompt="p")
        assert result.content == "Corrected."
        flagged = [e["claim"] for e in result.verification_log if e.get("is_valid") is False]
        assert flagged == ["Revenue grew 42% in 2023."]

    def test_a_skipped_claim_is_not_treated_as_a_hallucination(self) -> None:
        # A skipped entry has no is_valid key; reading it as invalid would
        # trigger a correction pass for a claim nobody checked.
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response("Draft."),
            _mock_response(self.CLAIMS),
            _mock_response("VERIFIED: True."),
        ]
        agent = AntiHallucinator(client, strictness=1.0, max_llm_calls=3)
        result = agent.generate(model="dummy", prompt="p")
        assert result.content == "Draft."
        assert result.budget_report()["exhausted"] is True

    def test_repeated_claims_each_get_their_own_log_entry(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response("Draft."),
            _mock_response("1. Same claim.\n2. Same claim."),
            _mock_response("VERIFIED: True."),
            _mock_response("VERIFIED: True."),
        ]
        agent = AntiHallucinator(client, strictness=1.0, max_llm_calls=4)
        result = agent.generate(model="dummy", prompt="p")
        assert [e["claim"] for e in result.verification_log] == ["Same claim.", "Same claim."]
