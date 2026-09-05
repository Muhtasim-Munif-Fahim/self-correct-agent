"""Tests for re-gating saved sessions against a verification policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_correct import sessions
from self_correct.cli import _build_parser, main
from self_correct.core import (
    AntiHallucinationResponse,
    VerificationPolicy,
    load_layered_policy,
)


def _save(tmp_path: Path, name: str, log: list[dict]) -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt="p",
        config={"model": "gpt-4o-mini"},
        result={
            "status": "flagged" if any(not e.get("is_valid") for e in log) else "verified",
            "content": "answer",
            "verification_log": log,
            "hallucinations_caught": [
                f"Claim '{e['claim']}' flagged: {e.get('critique', '')}"
                for e in log
                if isinstance(e, dict) and not e.get("is_valid")
            ],
        },
    )
    return path


def test_from_dict_recomputes_summaries_matching_persisted_payload() -> None:
    response = AntiHallucinationResponse(
        content="Earth is round and the sky is blue.",
        verification_log=[
            {
                "claim": "Earth is round",
                "is_valid": True,
                "critique": "",
                "evidence_sources": [{"title": "Wiki", "url": "https://a", "tool": "Wiki"}],
            },
            {
                "claim": "sky is green",
                "is_valid": False,
                "critique": "Statement is false and contradictory.",
                "evidence_sources": [],
            },
        ],
        hallucinations_caught=["Claim 'sky is green' flagged: false"],
    )
    data = response.to_dict()
    rebuilt = AntiHallucinationResponse.from_dict(data)

    assert rebuilt.content == data["content"]
    assert rebuilt.claim_summary() == data["claim_summary"]
    assert rebuilt.severity_summary() == data["severity_summary"]
    assert rebuilt.evidence_summary() == data["evidence_summary"]
    assert rebuilt.hallucination_density() == pytest.approx(data["hallucination_density"])


def test_from_dict_rejects_non_dict_payload() -> None:
    with pytest.raises(ValueError):
        AntiHallucinationResponse.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_from_dict_preserves_token_usage() -> None:
    data = {
        "content": "x",
        "verification_log": [],
        "hallucinations_caught": [],
        "token_usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    }
    rebuilt = AntiHallucinationResponse.from_dict(data)
    assert rebuilt.token_usage.prompt_tokens == 12
    assert rebuilt.token_usage.completion_tokens == 7
    assert rebuilt.token_usage.total_tokens == 19


def test_gate_sessions_passes_a_clean_session(tmp_path: Path) -> None:
    path = _save(tmp_path, "clean.json", [
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
    ])
    policy = VerificationPolicy()
    result = sessions.gate_sessions([path], policy)
    assert result["invalid"] == []
    assert result["results"] == [
        {
            "file": str(path),
            "status": "verified",
            "passed": True,
            "total_claims": 1,
            "verified_claims": 1,
            "flagged_claims": 0,
            "evidence_claims": 0,
            "reasons": [],
        }
    ]


def test_gate_sessions_fails_when_a_claim_is_flagged(tmp_path: Path) -> None:
    path = _save(tmp_path, "bad.json", [
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
        {"claim": "sky is green", "is_valid": False, "critique": "is false"},
    ])
    policy = VerificationPolicy(max_flagged_claims=0)
    result = sessions.gate_sessions([path], policy)
    row = result["results"][0]
    assert row["passed"] is False
    assert row["flagged_claims"] == 1
    assert row["reasons"]


def test_gate_sessions_reports_invalid_files(tmp_path: Path) -> None:
    good = _save(tmp_path, "good.json", [
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
    ])
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    result = sessions.gate_sessions([broken, good], VerificationPolicy())
    assert [row["file"] for row in result["results"]] == [str(good)]
    assert result["invalid"][0]["file"] == str(broken)
    assert "not valid JSON" in result["invalid"][0]["error"]


def test_gate_sessions_replays_merged_strict_policy(tmp_path: Path) -> None:
    path = _save(tmp_path, "s.json", [
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
        {"claim": "sky is green", "is_valid": False, "critique": "false"},
    ])
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"max_flagged_claims": 5}), encoding="utf-8")
    strict = tmp_path / "strict.json"
    strict.write_text(json.dumps({"max_flagged_claims": 0}), encoding="utf-8")

    policy, _conflicts = load_layered_policy([base, strict])
    result = sessions.gate_sessions([path], policy)
    assert result["results"][0]["passed"] is False


def test_cli_gate_prints_pass_and_fail_rows(tmp_path: Path, capsys) -> None:
    good = _save(tmp_path, "good.json", [
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
    ])
    bad = _save(tmp_path, "bad.json", [
        {"claim": "sky is green", "is_valid": False, "critique": "false"},
    ])
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"max_flagged_claims": 0}), encoding="utf-8")

    exit_code = main(["sessions-gate", str(tmp_path), "--policy", str(policy)])
    text = capsys.readouterr().out

    assert exit_code == 1
    assert "PASS" in text and "FAIL" in text
    assert f"Gate: 1 passed, 1 failed" in text
    assert str(good) in text and str(bad) in text


def test_cli_gate_json_output(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "bad.json", [
        {"claim": "sky is green", "is_valid": False, "critique": "false"},
    ])
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"max_flagged_claims": 0}), encoding="utf-8")

    exit_code = main(["sessions-gate", "--json", str(tmp_path), "--policy", str(policy)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["results"][0]["passed"] is False
    assert payload["results"][0]["reasons"]


def test_cli_gate_requires_policy(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "s.json", [{"claim": "x", "is_valid": True, "critique": ""}])

    exit_code = main(["sessions-gate", str(tmp_path)])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "--policy is required" in err


def test_cli_gate_returns_2_when_no_valid_sessions(tmp_path: Path, capsys) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"max_flagged_claims": 0}), encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("[]", encoding="utf-8")

    exit_code = main(["sessions-gate", str(broken), "--policy", str(policy)])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "No valid session files found." in err


def test_cli_gate_rejects_missing_policy_file(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "s.json", [{"claim": "x", "is_valid": True, "critique": ""}])

    exit_code = main(["sessions-gate", str(tmp_path), "--policy", str(tmp_path / "nope.json")])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "sessions-gate:" in err
