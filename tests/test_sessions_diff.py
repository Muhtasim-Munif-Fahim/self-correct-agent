"""Tests for the sessions-diff CLI subcommand."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from self_correct import sessions
from self_correct.cli import main


def _save(tmp_path: Path, name: str, claims: list[dict]) -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt="p",
        config={"model": "gpt-4o-mini"},
        result={
            "status": "verified",
            "content": "",
            "verification_log": claims,
            "hallucinations_caught": [],
        },
    )
    return path


def test_sessions_diff_json_round_trip(tmp_path: Path) -> None:
    baseline = _save(tmp_path, "baseline.json", claims=[
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
        {"claim": "Sky is green", "is_valid": True, "critique": ""},
    ])
    current = _save(tmp_path, "current.json", claims=[
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
        {"claim": "Sky is green", "is_valid": False, "critique": "It is blue."},
        {"claim": "Pluto is a planet", "is_valid": False, "critique": "Reclassified."},
    ])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["sessions-diff", str(baseline), str(current), "--json"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "sky is green" in payload["regressed"]
    assert "pluto is a planet" in payload["added"]
    assert payload["unchanged_verified"] == 1
    assert payload["baseline_claims"] == 2
    assert payload["current_claims"] == 3


def test_sessions_diff_text_output_lists_categories(tmp_path: Path) -> None:
    baseline = _save(tmp_path, "baseline.json", claims=[
        {"claim": "A", "is_valid": True, "critique": ""},
        {"claim": "B", "is_valid": False, "critique": ""},
    ])
    current = _save(tmp_path, "current.json", claims=[
        {"claim": "A", "is_valid": True, "critique": ""},
        {"claim": "B", "is_valid": True, "critique": "fixed"},
        {"claim": "C", "is_valid": False, "critique": ""},
    ])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["sessions-diff", str(baseline), str(current)])
    assert rc == 0
    text = buf.getvalue()
    assert "Baseline claims: 2" in text
    assert "Current claims: 3" in text
    assert "Resolved" in text
    assert "Added claims" in text
    assert "Unchanged verified" in text


def test_sessions_diff_rejects_missing_max_listed(tmp_path: Path) -> None:
    baseline = _save(tmp_path, "baseline.json", claims=[{"claim": "x", "is_valid": True, "critique": ""}])
    current = _save(tmp_path, "current.json", claims=[{"claim": "x", "is_valid": True, "critique": ""}])
    buf_err = io.StringIO()
    with redirect_stdout(io.StringIO()):
        with redirect_stdout(buf_err):
            import sys
            sys.stderr = buf_err
            rc = main(["sessions-diff", str(baseline), str(current), "--max-listed", "0"])
    assert rc == 2


def test_sessions_diff_returns_2_for_missing_file(tmp_path: Path) -> None:
    baseline = _save(tmp_path, "baseline.json", claims=[{"claim": "x", "is_valid": True, "critique": ""}])
    rc = main(["sessions-diff", str(baseline), str(tmp_path / "missing.json")])
    assert rc == 2


def test_sessions_diff_truncates_long_lists(tmp_path: Path) -> None:
    many = [{"claim": f"claim_{i}", "is_valid": False, "critique": ""} for i in range(50)]
    baseline = _save(tmp_path, "baseline.json", claims=[])
    current = _save(tmp_path, "current.json", claims=many)
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["sessions-diff", str(baseline), str(current), "--max-listed", "3"])
    text = buf.getvalue()
    assert "and 47 more" in text