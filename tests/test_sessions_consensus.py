"""Tests for scoring per-claim verification consensus across saved sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_correct import sessions
from self_correct.cli import _build_parser, main


def _save(tmp_path: Path, name: str, claims: list[tuple[str, bool]]) -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt="p",
        config={"model": "gpt-4o-mini"},
        result={
            "status": "verified" if all(v for _, v in claims) else "flagged",
            "content": "answer",
            "verification_log": [
                {"claim": claim, "is_valid": is_valid, "critique": ""}
                for claim, is_valid in claims
            ],
            "hallucinations_caught": [],
        },
    )
    return path


def test_claim_consensus_stable_claim_across_sessions(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", [("Earth is round", True)])
    _save(tmp_path, "b.json", [("Earth is round", True)])
    result = sessions.claim_consensus([tmp_path])
    assert result["scanned"] == 2
    assert len(result["claims"]) == 1
    claim = result["claims"][0]
    assert claim["claim"] == "Earth is round"
    assert claim["verified"] == 2
    assert claim["flagged"] == 0
    assert claim["consensus_ratio"] == 1.0
    assert claim["stable"] is True


def test_claim_consensus_flip_flop_claim(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", [("sky is blue", True)])
    _save(tmp_path, "b.json", [("sky is blue", False)])
    result = sessions.claim_consensus([tmp_path])
    claim = result["claims"][0]
    assert claim["verified"] == 1
    assert claim["flagged"] == 1
    assert claim["consensus_ratio"] == 0.5
    assert claim["stable"] is False
    assert result["totals"]["flip_flop_claims"] == 1
    assert result["totals"]["stable_claims"] == 0


def test_claim_consensus_min_sessions_filters_single_occurrences(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", [("Earth is round", True)])
    _save(tmp_path, "b.json", [("Earth is round", True), ("mars red", True)])
    result = sessions.claim_consensus([tmp_path], min_sessions=2)
    claims = {c["claim"] for c in result["claims"]}
    assert "Earth is round" in claims
    assert "mars red" not in claims


def test_claim_consensus_normalizes_claim_text(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", [("  Earth Is Round  ", True)])
    _save(tmp_path, "b.json", [("earth is round", True)])
    result = sessions.claim_consensus([tmp_path])
    assert len(result["claims"]) == 1
    assert result["claims"][0]["verified"] == 2


def test_claim_consensus_reports_invalid_files(tmp_path: Path) -> None:
    good = _save(tmp_path, "good.json", [("Earth is round", True)])
    _save(tmp_path, "good2.json", [("Earth is round", True)])
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    result = sessions.claim_consensus([good, broken])
    assert result["invalid"][0]["file"] == str(broken)
    assert "not valid JSON" in result["invalid"][0]["error"]
    assert result["scanned"] == 1


def test_claim_consensus_empty_log(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", [])
    _save(tmp_path, "b.json", [])
    result = sessions.claim_consensus([tmp_path])
    assert result["claims"] == []
    assert result["totals"]["unique_claims"] == 0
    assert result["totals"]["average_consensus_ratio"] == 0.0


def test_claim_consensus_rejects_bad_min_sessions() -> None:
    with pytest.raises(ValueError):
        sessions.claim_consensus([], min_sessions=0)
    with pytest.raises(ValueError):
        sessions.claim_consensus([], min_sessions=-1)


def test_cli_consensus_json(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [("Earth is round", True)])
    _save(tmp_path, "b.json", [("Earth is round", True)])
    rc = main(["sessions-consensus", "--json", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["scanned"] == 2
    assert payload["totals"]["unique_claims"] == 1


def test_cli_consensus_text(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [("Earth is round", True)])
    _save(tmp_path, "b.json", [("Earth is round", True)])
    rc = main(["sessions-consensus", str(tmp_path), "--max-listed", "50"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 unique claim(s)" in out
    assert "STABLE" in out


def test_cli_consensus_flip_only(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [("Earth is round", True), ("sky is blue", True)])
    _save(tmp_path, "b.json", [("Earth is round", True), ("sky is blue", False)])
    rc = main(["sessions-consensus", "--flip-only", "--json", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    flip = payload["claims"]
    assert len(flip) == 1
    assert flip[0]["claim"] == "sky is blue"


def test_cli_consensus_returns_2_when_no_valid_sessions(tmp_path: Path, capsys) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("[]", encoding="utf-8")
    rc = main(["sessions-consensus", str(broken)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "No valid session files found." in err


def test_cli_consensus_rejects_min_sessions_below_one(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", [("Earth is round", True)])
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["sessions-consensus", "--min-sessions", "0", str(tmp_path)])


def test_subcommand_is_registered() -> None:
    args = _build_parser().parse_args(["sessions-consensus", "dir"])
    assert args.paths == ["dir"]
    assert args.min_sessions == 2
    assert args.flip_only is False
    assert args.json is False
