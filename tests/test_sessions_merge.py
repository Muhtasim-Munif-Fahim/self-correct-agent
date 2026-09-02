"""Tests for the sessions-merge feature."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_correct import sessions


def _save(tmp_path: Path, name: str, *, prompt: str, claims: list[dict]) -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt=prompt,
        config={"model": "gpt-4o-mini"},
        result={"status": "verified", "content": "", "verification_log": claims, "hallucinations_caught": []},
    )
    return path


def test_merge_dedupes_claims_by_normalized_text(tmp_path) -> None:
    a = _save(tmp_path, "a.json", prompt="p", claims=[
        {"claim": "Earth orbits the Sun", "is_valid": True, "critique": ""},
        {"claim": "Mercury is the closest planet", "is_valid": True, "critique": ""},
    ])
    b = _save(tmp_path, "b.json", prompt="p", claims=[
        {"claim": "earth orbits the sun", "is_valid": True, "critique": ""},
        {"claim": "Pluto is a planet", "is_valid": False, "critique": "Reclassified in 2006."},
    ])
    merged = sessions.merge_sessions([a, b])
    claims = merged["result"]["verification_log"]
    assert len(claims) == 3
    texts = [entry["claim"] for entry in claims]
    assert "Earth orbits the Sun" in texts
    assert "Pluto is a planet" in texts
    assert merged["result"]["hallucinations_caught"] == ["Pluto is a planet"]


def test_merge_keeps_flagged_when_keep_flag_and_verified_appears(tmp_path) -> None:
    a = _save(tmp_path, "a.json", prompt="p", claims=[
        {"claim": "Sky is green", "is_valid": True, "critique": ""},
    ])
    b = _save(tmp_path, "b.json", prompt="p", claims=[
        {"claim": "Sky is green", "is_valid": False, "critique": "It is blue."},
    ])
    merged = sessions.merge_sessions([a, b], keep="flag", on_conflict="keep")
    assert len(merged["result"]["verification_log"]) == 1
    assert merged["result"]["verification_log"][0]["is_valid"] is False
    assert len(merged["conflicts"]) == 1


def test_merge_keeps_verified_when_keep_verify_and_conflict_skipped(tmp_path) -> None:
    a = _save(tmp_path, "a.json", prompt="p", claims=[
        {"claim": "Water boils at 100C", "is_valid": True, "critique": ""},
    ])
    b = _save(tmp_path, "b.json", prompt="p", claims=[
        {"claim": "Water boils at 100C", "is_valid": False, "critique": "Varies with pressure."},
    ])
    merged = sessions.merge_sessions([a, b], keep="verify", on_conflict="skip")
    # Conflict on_conflict=skip drops the disputed claim entirely.
    assert merged["result"]["verification_log"] == []
    assert merged["conflicts"]


def test_merge_on_conflict_keep_keeps_winning_verdict(tmp_path) -> None:
    a = _save(tmp_path, "a.json", prompt="p", claims=[
        {"claim": "Sharks are fish", "is_valid": False, "critique": "Wrong"},
    ])
    b = _save(tmp_path, "b.json", prompt="p", claims=[
        {"claim": "Sharks are fish", "is_valid": True, "critique": ""},
    ])
    merged = sessions.merge_sessions([a, b], keep="verify", on_conflict="keep")
    assert len(merged["result"]["verification_log"]) == 1
    assert merged["result"]["verification_log"][0]["is_valid"] is True


def test_merge_keeps_any_picks_first_verdict_seen(tmp_path) -> None:
    a = _save(tmp_path, "a.json", prompt="p", claims=[
        {"claim": "Oxygen is diatomic", "is_valid": True, "critique": ""},
    ])
    b = _save(tmp_path, "b.json", prompt="p", claims=[
        {"claim": "Oxygen is diatomic", "is_valid": False, "critique": ""},
    ])
    merged = sessions.merge_sessions([a, b], keep="any", on_conflict="keep")
    assert merged["result"]["verification_log"][0]["is_valid"] is True


def test_merge_rejects_invalid_keep_value() -> None:
    with pytest.raises(ValueError, match="keep must be"):
        sessions.merge_sessions([], keep="bogus")


def test_merge_rejects_invalid_on_conflict_value() -> None:
    with pytest.raises(ValueError, match="on_conflict must be"):
        sessions.merge_sessions([], on_conflict="bogus")


def test_merge_records_invalid_files(tmp_path) -> None:
    good = _save(tmp_path, "good.json", prompt="p", claims=[
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
    ])
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    merged = sessions.merge_sessions([good, broken])
    assert len(merged["source_files"]) == 1
    assert len(merged["invalid"]) == 1
    assert merged["invalid"][0]["file"] == str(broken)


def test_merge_returns_empty_when_no_valid_files(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    merged = sessions.merge_sessions([broken])
    assert merged["source_files"] == []
    assert merged["result"]["verification_log"] == []
    assert merged["result"]["status"] == "unknown"


def test_merge_skips_non_verdict_entries(tmp_path) -> None:
    a = _save(tmp_path, "a.json", prompt="p", claims=[
        {"phase": "extract", "info": "ignored"},
        {"claim": "Sky is blue", "is_valid": True, "critique": ""},
        {"claim": "  ", "is_valid": False, "critique": ""},
    ])
    merged = sessions.merge_sessions([a])
    assert len(merged["result"]["verification_log"]) == 1


def test_cli_sessions_merge_writes_output_file(tmp_path, capsys, monkeypatch) -> None:
    from self_correct import cli as cli_module

    a = _save(tmp_path, "a.json", prompt="p", claims=[
        {"claim": "Earth orbits Sun", "is_valid": True, "critique": ""},
    ])
    b = _save(tmp_path, "b.json", prompt="p", claims=[
        {"claim": "Pluto is a planet", "is_valid": False, "critique": "Reclassified."},
    ])
    out = tmp_path / "merged.json"
    parser = cli_module._build_parser()
    args = parser.parse_args(["sessions-merge", str(a), str(b), "-o", str(out), "--json"])
    rc = cli_module._cmd_sessions_merge(args)
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == sessions.SESSION_SCHEMA_VERSION
    summary = json.loads(capsys.readouterr().out)
    assert summary["conflicts"] == []