"""Tests for hallucination-density scoring on verification results."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from self_correct.cli import _format_hallucination_density, _render_text_report
from self_correct.core import AntiHallucinationResponse
from self_correct.junit import result_to_junit_xml
from self_correct.review import render_session_review, render_session_review_with_counts


def _response(*, content: str, claims) -> AntiHallucinationResponse:
    log = [
        {
            "claim": claim,
            "is_valid": is_valid,
            "critique": "" if is_valid else "unsupported",
        }
        for claim, is_valid in claims
    ]
    return AntiHallucinationResponse(
        content=content,
        verification_log=log,
        hallucinations_caught=[
            f"Claim '{claim}' flagged: unsupported"
            for claim, is_valid in claims
            if not is_valid
        ],
    )


def test_zero_claims_scores_zero_rate_and_zero_per_words() -> None:
    report = _response(content="", claims=[]).hallucination_density_report()
    assert report == {
        "total_claims": 0,
        "flagged_claims": 0,
        "claim_rate": 0.0,
        "word_count": 0,
        "per_words": 100,
        "per_words_density": 0.0,
    }


def test_all_verified_claims_have_zero_claim_rate() -> None:
    response = _response(
        content="one two three four five",
        claims=[("sky is blue", True), ("water is wet", True)],
    )
    report = response.hallucination_density_report()
    assert report["total_claims"] == 2
    assert report["flagged_claims"] == 0
    assert report["claim_rate"] == 0.0
    assert report["word_count"] == 5
    assert report["per_words_density"] == 0.0


def test_mixed_flagged_claims_report_flagged_over_total() -> None:
    response = _response(
        content="word " * 100,
        claims=[
            ("sky is blue", True),
            ("moon is cheese", False),
            ("water boils at 100C", True),
            ("earth is a cube", False),
        ],
    )
    report = response.hallucination_density_report()
    assert report["total_claims"] == 4
    assert report["flagged_claims"] == 2
    assert report["claim_rate"] == 0.5
    assert report["word_count"] == 100
    assert report["per_words_density"] == 2.0


def test_all_flagged_claims_have_unit_claim_rate() -> None:
    report = _response(
        content="a b c d",
        claims=[("x", False), ("y", False)],
    ).hallucination_density_report()
    assert report["claim_rate"] == 1.0
    assert report["flagged_claims"] == 2
    assert report["total_claims"] == 2


def test_report_ignores_phase_metadata_entries() -> None:
    response = AntiHallucinationResponse(
        content="draft text here",
        verification_log=[
            {"phase": "extraction", "warning": "none"},
            {"claim": "kept", "is_valid": True, "critique": ""},
            {"claim": "bad", "is_valid": False, "critique": "false"},
            {"skipped_by_budget": True, "phase": "correction"},
        ],
        hallucinations_caught=["Claim 'bad' flagged: false"],
    )
    report = response.hallucination_density_report()
    assert report["total_claims"] == 2
    assert report["flagged_claims"] == 1
    assert report["claim_rate"] == 0.5


def test_per_words_density_is_optional_and_scales() -> None:
    response = _response(
        content="word " * 100,
        claims=[("x", False)],
    )
    default = response.hallucination_density_report()
    scaled = response.hallucination_density_report(per_words=50)
    assert default["per_words"] == 100
    assert default["per_words_density"] == 1.0
    assert scaled["per_words"] == 50
    assert scaled["per_words_density"] == 0.5
    assert scaled["claim_rate"] == default["claim_rate"] == 1.0


def test_report_rejects_non_positive_chunk() -> None:
    response = _response(content="some words", claims=[("x", True)])
    with pytest.raises(ValueError, match="per_words"):
        response.hallucination_density_report(per_words=0)


def test_format_line_includes_claim_rate_and_per_words() -> None:
    text = _response(
        content="word " * 100,
        claims=[("a", True), ("b", False)],
    ).format_hallucination_density()
    assert "50.0% of claims" in text
    assert "(1/2 flagged)" in text
    assert "1.00 per 100 words" in text


def test_report_is_serialized_on_to_dict_and_to_json() -> None:
    response = _response(
        content="hello world",
        claims=[("a", True), ("b", False)],
    )
    payload = json.loads(response.to_json())
    assert payload["hallucination_density"] == round(response.hallucination_density(), 3)
    assert payload["hallucination_density_report"] == response.hallucination_density_report()
    assert payload["hallucination_density_report"]["claim_rate"] == 0.5


def test_from_dict_recomputes_the_density_report() -> None:
    response = _response(
        content="one two three four",
        claims=[("a", True), ("b", False)],
    )
    rebuilt = AntiHallucinationResponse.from_dict(response.to_dict())
    assert rebuilt.hallucination_density_report() == response.hallucination_density_report()


def test_markdown_and_html_surface_the_formatted_density() -> None:
    response = _response(
        content="word " * 50,
        claims=[("a", True), ("b", False), ("c", True)],
    )
    formatted = response.format_hallucination_density()
    markdown = response.to_markdown()
    html = response.to_html()
    assert f"**Hallucination density**: {formatted}" in markdown
    assert "Hallucination density" in html
    assert "33.3% of claims" in html
    assert "(1/3 flagged)" in html


def test_cli_text_report_includes_density() -> None:
    response = _response(
        content="word " * 20,
        claims=[("ok", True), ("bad", False)],
    )
    args = SimpleNamespace(model="gpt-4o-mini", quiet_ok=True)
    text = _render_text_report(args, response)
    assert "Hallucination density:" in text
    assert "50.0% of claims" in text
    assert "(1/2 flagged)" in text


def test_cli_formats_density_from_duck_typed_results() -> None:
    result = SimpleNamespace(
        content="a b c d e",
        verification_log=[
            {"claim": "a", "is_valid": True},
            {"claim": "b", "is_valid": False},
        ],
        hallucinations_caught=["b"],
    )
    text = _format_hallucination_density(result)
    assert "50.0% of claims" in text
    assert "(1/2 flagged)" in text


def test_review_headline_includes_density() -> None:
    session = {
        "prompt": "p",
        "config": {},
        "result": {
            "content": "word " * 10,
            "verification_log": [
                {"claim": "a", "is_valid": True, "critique": ""},
                {"claim": "b", "is_valid": False, "critique": "false"},
            ],
        },
    }
    markdown = render_session_review(session)
    assert "Hallucination density" in markdown
    assert "50.00% of claims" in markdown
    payload = render_session_review_with_counts(session)
    assert payload["counts"]["hallucination_density"]["claim_rate"] == 0.5
    assert payload["counts"]["hallucination_density"]["flagged_claims"] == 1


def test_junit_export_attaches_density_properties() -> None:
    xml_text = result_to_junit_xml({
        "content": "word " * 100,
        "verification_log": [
            {"claim": "a", "is_valid": True, "critique": ""},
            {"claim": "b", "is_valid": False, "critique": "false"},
        ],
        "hallucinations_caught": ["Claim 'b' flagged: false"],
    })
    assert 'name="hallucination_density.claim_rate"' in xml_text
    assert 'value="0.5"' in xml_text
    assert 'name="hallucination_density.per_words_density"' in xml_text
