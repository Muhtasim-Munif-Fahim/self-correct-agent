"""Tests for rendering verification reports as self-contained HTML."""

from __future__ import annotations

from self_correct.cli import _build_parser, _detect_output_format
from self_correct.core import AntiHallucinationResponse, TokenUsage


def test_html_is_self_contained() -> None:
    response = AntiHallucinationResponse(content="clean answer")
    html = response.to_html()
    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html and "</html>" in html
    assert "<style>" in html
    assert "clean answer" in html


def test_html_escapes_content_that_looks_like_markup() -> None:
    response = AntiHallucinationResponse(content="<script>alert(1)</script>")
    html = response.to_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_html_reports_tokens_and_duration() -> None:
    response = AntiHallucinationResponse(
        content="text",
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        elapsed_seconds=1.25,
    )
    html = response.to_html()
    assert "30" in html
    assert "1.25s" in html


def test_html_lists_flagged_claims_with_severity() -> None:
    response = AntiHallucinationResponse(
        content="text",
        hallucinations_caught=["The figure is fabricated"],
    )
    html = response.to_html()
    assert '<span class="flagged">[critical]</span>' in html
    assert "The figure is fabricated" in html


def test_html_log_is_only_included_when_requested() -> None:
    response = AntiHallucinationResponse(
        content="text",
        verification_log=[
            {"claim": "sky is blue", "is_valid": True},
            {"claim": "moon is cheese", "is_valid": False, "critique": "Moon is rock."},
        ],
    )
    assert "Verification Log" not in response.to_html()
    html = response.to_html(include_log=True)
    assert "Verification Log" in html
    assert "sky is blue" in html
    assert "moon is cheese" in html
    assert "Moon is rock." in html


def test_parser_accepts_html_output_format() -> None:
    args = _build_parser().parse_args(
        ["verify", "--prompt", "p", "--output-format", "html"]
    )
    assert args.output_format == "html"


def test_resume_parser_accepts_html_output_format() -> None:
    args = _build_parser().parse_args(
        ["resume", "session.json", "--output-format", "html"]
    )
    assert args.output_format == "html"


def test_html_extension_is_detected() -> None:
    assert _detect_output_format("report.html", None) == "html"
    assert _detect_output_format("report.htm", None) == "html"
    assert _detect_output_format("report.md", None) == "markdown"
    assert _detect_output_format("report.json", None) == "json"
