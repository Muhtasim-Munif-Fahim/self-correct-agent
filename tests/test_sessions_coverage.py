"""Tests for verification coverage rollups across saved session files."""

from __future__ import annotations

import json
from pathlib import Path

from self_correct import sessions
from self_correct.cli import _build_parser, main


def _save(tmp_path: Path, name: str, log: list, status: str = "verified") -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt="p",
        config={"model": "gpt-4o-mini"},
        result={
            "status": status,
            "content": "answer",
            "verification_log": log,
            "hallucinations_caught": [
                f"Claim '{e.get('claim')}' flagged: {e.get('critique', '')}"
                for e in log
                if isinstance(e, dict) and "claim" in e and e.get("is_valid") is False
            ],
        },
    )
    return path


def test_verification_coverage_counts_verified_flagged_skipped(tmp_path: Path) -> None:
    path = _save(tmp_path, "mixed.json", [
        {"claim": "Earth is round", "is_valid": True},
        {"claim": "sky is green", "is_valid": False, "critique": "false"},
    ])
    row = sessions.verification_coverage([path])["sessions"][0]
    assert row["verified"] == 1
    assert row["flagged"] == 1
    assert row["skipped"] == 0
    assert row["claims"] == 2
    assert row["coverage_ratio"] == 1.0
    assert row["budget_exhausted"] is False


def test_verification_coverage_flags_budget_skipped_claims(tmp_path: Path) -> None:
    path = _save(tmp_path, "budget.json", [
        {"claim": "a", "is_valid": True},
        {"claim": "b", "is_valid": True},
        {"claim": "c", "skipped_by_budget": True},
        {"phase": "correction", "skipped_by_budget": True},
    ])
    row = sessions.verification_coverage([path])["sessions"][0]
    assert row["verified"] == 2
    assert row["flagged"] == 0
    assert row["skipped"] == 1
    assert row["claims"] == 3
    assert row["coverage_ratio"] == round(2 / 3, 3)
    assert row["budget_exhausted"] is True


def test_verification_coverage_empty_log_is_full_coverage(tmp_path: Path) -> None:
    path = _save(tmp_path, "empty.json", [])
    row = sessions.verification_coverage([path])["sessions"][0]
    assert row["claims"] == 0
    assert row["coverage_ratio"] == 1.0
    assert row["budget_exhausted"] is False


def test_verification_coverage_ignores_phase_only_entries(tmp_path: Path) -> None:
    path = _save(tmp_path, "phases.json", [
        {"phase": "extraction", "warning": "No claims extracted."},
        {"phase": "budget", "skipped_by_budget": False},
        {"claim": "Earth is round", "is_valid": True},
    ])
    row = sessions.verification_coverage([path])["sessions"][0]
    assert row["claims"] == 1
    assert row["verified"] == 1
    assert row["skipped"] == 0
    assert row["budget_exhausted"] is False


def test_verification_coverage_aggregates_totals(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", [
        {"claim": "Earth is round", "is_valid": True},
        {"claim": "sky is green", "is_valid": False, "critique": "false"},
    ])
    _save(tmp_path, "b.json", [
        {"claim": "a", "is_valid": True},
        {"claim": "b", "is_valid": True},
        {"claim": "c", "skipped_by_budget": True},
        {"phase": "correction", "skipped_by_budget": True},
    ])
    _save(tmp_path, "c.json", [])
    agg = sessions.verification_coverage([tmp_path])
    totals = agg["totals"]
    assert totals["sessions"] == 3
    assert totals["claims"] == 5
    assert totals["verified"] == 3
    assert totals["flagged"] == 1
    assert totals["skipped"] == 1
    assert totals["coverage_ratio"] == round(4 / 5, 3)
    assert totals["budget_exhausted_sessions"] == 1
    assert agg["invalid"] == []


def test_verification_coverage_reports_invalid_files(tmp_path: Path) -> None:
    good = _save(tmp_path, "good.json", [
        {"claim": "Earth is round", "is_valid": True},
    ])
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    agg = sessions.verification_coverage([broken, good])
    assert [row["file"] for row in agg["sessions"]] == [str(good)]
    assert agg["totals"]["sessions"] == 1
    assert agg["invalid"][0]["file"] == str(broken)
    assert "not valid JSON" in agg["invalid"][0]["error"]


def test_cli_sessions_coverage_json(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [
        {"claim": "Earth is round", "is_valid": True},
        {"claim": "c", "skipped_by_budget": True},
    ])
    rc = main(["sessions-coverage", "--json", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["totals"]["sessions"] == 1
    assert payload["totals"]["skipped"] == 1
    assert payload["totals"]["budget_exhausted_sessions"] == 1


def test_cli_sessions_coverage_text(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [
        {"claim": "Earth is round", "is_valid": True},
        {"claim": "sky is green", "is_valid": False, "critique": "false"},
    ])
    _save(tmp_path, "b.json", [
        {"claim": "a", "is_valid": True},
        {"claim": "c", "skipped_by_budget": True},
    ])
    rc = main(["sessions-coverage", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Coverage across 2 session(s)" in out
    assert "BUDGET" in out
    assert "ok" in out


def test_cli_sessions_coverage_returns_2_when_no_valid(tmp_path: Path, capsys) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("[]", encoding="utf-8")
    rc = main(["sessions-coverage", str(broken)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "No valid session files found." in err


def test_subcommand_is_registered() -> None:
    args = _build_parser().parse_args(["sessions-coverage", "dir"])
    assert args.paths == ["dir"]
    assert args.json is False
