"""Tests for estimating USD cost across saved session files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_correct import sessions
from self_correct.cli import _build_parser, main
from self_correct.core import MODEL_PRICING


def _save(tmp_path: Path, name: str, model: str, prompt_tokens: int, completion_tokens: int) -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt="p",
        config={"model": model},
        result={
            "status": "verified",
            "content": "answer",
            "verification_log": [],
            "hallucinations_caught": [],
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    )
    return path


def test_estimate_session_cost_known_model() -> None:
    session = {
        "config": {"model": "gpt-4o-mini"},
        "result": {
            "token_usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
        },
    }
    estimate = sessions.estimate_session_cost(session)
    pr, cr = MODEL_PRICING["gpt-4o-mini"]
    expected = (1000 / 1_000_000) * pr + (1000 / 1_000_000) * cr
    assert estimate["model"] == "gpt-4o-mini"
    assert estimate["cost_usd"] == round(expected, 6)
    assert estimate["cost_unknown"] is False


def test_estimate_session_cost_unknown_model() -> None:
    session = {
        "config": {"model": "claude-unknown"},
        "result": {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    }
    estimate = sessions.estimate_session_cost(session)
    assert estimate["cost_unknown"] is True
    assert estimate["cost_usd"] is None
    assert estimate["prompt_tokens"] == 10
    assert estimate["completion_tokens"] == 5


def test_estimate_session_cost_model_override() -> None:
    session = {
        "config": {"model": "claude-unknown"},
        "result": {"token_usage": {"prompt_tokens": 1000, "completion_tokens": 1000}},
    }
    estimate = sessions.estimate_session_cost(session, model="gpt-4o-mini")
    assert estimate["cost_unknown"] is False
    assert estimate["model"] == "gpt-4o-mini"


def test_estimate_session_cost_missing_token_usage() -> None:
    session = {"config": {"model": "gpt-4o-mini"}, "result": {}}
    estimate = sessions.estimate_session_cost(session)
    assert estimate["prompt_tokens"] == 0
    assert estimate["completion_tokens"] == 0
    assert estimate["cost_usd"] == 0.0


def test_cost_sessions_aggregates(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", "gpt-4o-mini", 1000, 1000)
    _save(tmp_path, "b.json", "gpt-4o", 1000, 1000)
    result = sessions.cost_sessions([tmp_path])
    assert result["scanned"] == 2
    assert result["invalid"] == []
    assert result["totals"]["sessions"] == 2
    assert result["totals"]["prompt_tokens"] == 2000
    assert result["totals"]["completion_tokens"] == 2000
    assert result["totals"]["total_tokens"] == 4000
    assert result["totals"]["cost_usd"] > 0
    assert result["totals"]["unknown_model_sessions"] == 0


def test_cost_sessions_reports_unknown_models(tmp_path: Path) -> None:
    _save(tmp_path, "known.json", "gpt-4o-mini", 1000, 1000)
    _save(tmp_path, "unknown.json", "claude-weird", 500, 500)
    result = sessions.cost_sessions([tmp_path])
    assert result["totals"]["unknown_model_sessions"] == 1
    assert result["totals"]["cost_usd"] > 0


def test_cost_sessions_reports_invalid_files(tmp_path: Path) -> None:
    good = _save(tmp_path, "good.json", "gpt-4o-mini", 100, 100)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    result = sessions.cost_sessions([good, broken])
    assert [row["file"] for row in result["sessions"]] == [str(good)]
    assert result["invalid"][0]["file"] == str(broken)
    assert "not valid JSON" in result["invalid"][0]["error"]


def test_cli_sessions_cost_json(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", "gpt-4o-mini", 1000, 1000)
    rc = main(["sessions-cost", "--json", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["scanned"] == 1
    assert payload["totals"]["cost_usd"] > 0


def test_cli_sessions_cost_text(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", "gpt-4o-mini", 1000, 1000)
    _save(tmp_path, "b.json", "claude-unknown", 500, 500)
    rc = main(["sessions-cost", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tokens (gpt-4o-mini)" in out
    assert "tokens (claude-unknown)" in out
    assert "unknown (no published rate)" in out
    assert "lower bound" in out
    assert "Total across 2 session(s)" in out


def test_cli_sessions_cost_model_override(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", "claude-unknown", 1000, 1000)
    rc = main(["sessions-cost", "--model", "gpt-4o-mini", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tokens (gpt-4o-mini)" in out


def test_cli_sessions_cost_returns_2_when_no_valid_sessions(tmp_path: Path, capsys) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("[]", encoding="utf-8")
    rc = main(["sessions-cost", str(broken)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "No valid session files found." in err


def test_subcommand_is_registered() -> None:
    args = _build_parser().parse_args(["sessions-cost", "dir"])
    assert args.paths == ["dir"]
    assert args.model is None
    assert args.json is False
