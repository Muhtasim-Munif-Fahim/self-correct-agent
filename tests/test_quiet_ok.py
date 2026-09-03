"""Tests for the verify --quiet-ok flag."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from types import SimpleNamespace

from self_correct import cli as cli_module


def _fake_result(*, claims: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        token_usage=SimpleNamespace(
            total_tokens=100, prompt_tokens=70, completion_tokens=30,
        ),
        elapsed_seconds=0.42,
        status="verified",
        content="final output",
        hallucinations_caught=[c["claim"] for c in claims if not c.get("is_valid")],
        verification_log=list(claims),
    )


def test_quiet_ok_omits_ok_claims_block() -> None:
    args = SimpleNamespace(
        model="gpt-4o-mini", tools=[],
        strictness=1, no_cache=True, cache_ttl=None, max_retries=0,
        retry_backoff=0.0, max_calls=None, content_checks=None,
        model_draft=None, model_extract=None, model_verify=None, model_correct=None,
        cache_file=None, save_session=None, output=None, output_format=None,
        include_log=True, include_ok=True,
        quiet_ok=True,
    )
    result = _fake_result(claims=[
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
        {"claim": "Sky is green", "is_valid": False, "critique": "blue"},
    ])
    text = cli_module._render_text_report(args, result)
    assert "OK Claims" not in text
    assert "Sky is green" in text
    assert "Earth is round" not in text


def test_default_prints_ok_claims_block() -> None:
    args = SimpleNamespace(
        model="gpt-4o-mini", tools=[],
        strictness=1, no_cache=True, cache_ttl=None, max_retries=0,
        retry_backoff=0.0, max_calls=None, content_checks=None,
        model_draft=None, model_extract=None, model_verify=None, model_correct=None,
        cache_file=None, save_session=None, output=None, output_format=None,
        include_log=True, include_ok=True,
        quiet_ok=False,
    )
    result = _fake_result(claims=[
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
        {"claim": "Sky is green", "is_valid": False, "critique": "blue"},
    ])
    text = cli_module._render_text_report(args, result)
    assert "OK Claims" in text
    assert "Earth is round" in text


def test_quiet_ok_handles_no_ok_claims() -> None:
    args = SimpleNamespace(
        model="gpt-4o-mini", tools=[],
        strictness=1, no_cache=True, cache_ttl=None, max_retries=0,
        retry_backoff=0.0, max_calls=None, content_checks=None,
        model_draft=None, model_extract=None, model_verify=None, model_correct=None,
        cache_file=None, save_session=None, output=None, output_format=None,
        include_log=True, include_ok=True,
        quiet_ok=True,
    )
    result = _fake_result(claims=[
        {"claim": "Sky is green", "is_valid": False, "critique": "blue"},
    ])
    text = cli_module._render_text_report(args, result)
    assert "OK Claims" not in text
    assert "Sky is green" in text