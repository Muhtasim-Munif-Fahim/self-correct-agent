"""Tests for exporting saved sessions as JSONL claim verdicts."""

from __future__ import annotations

import json

from self_correct.cli import _build_parser, _cmd_export_jsonl
from self_correct.jsonlreport import result_to_jsonl
from self_correct.sessions import save_session


def _result(*claims):
    return {
        "content": "answer",
        "verification_log": [
            {
                "claim": claim,
                "is_valid": is_valid,
                "critique": critique,
                "evidence_used": evidence,
                "cached": cached,
            }
            for claim, is_valid, critique, evidence, cached in claims
        ],
    }


def _records(text: str):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_each_verdict_becomes_one_json_object() -> None:
    text = result_to_jsonl(_result(
        ("sky is blue", True, "", False, False),
        ("moon is cheese", False, "Moon is rock.", True, False),
    ))
    records = _records(text)
    assert [record["claim"] for record in records] == ["sky is blue", "moon is cheese"]
    assert records[0]["verdict"] == "verified"
    assert records[1]["verdict"] == "flagged"


def test_booleans_stay_typed_in_json() -> None:
    records = _records(result_to_jsonl(_result(
        ("x", False, "Statement is false.", True, True),
    )))
    assert records[0]["evidence_used"] is True
    assert records[0]["cached"] is True


def test_flagged_rows_carry_severity_and_critique() -> None:
    records = _records(result_to_jsonl(_result(
        ("x", False, "Statement is false.", False, False),
    )))
    assert records[0]["severity"] == "critical"
    assert records[0]["critique"] == "Statement is false."


def test_verified_rows_have_empty_severity() -> None:
    records = _records(result_to_jsonl(_result(
        ("y", True, "", True, True),
    )))
    assert records[0]["severity"] == ""


def test_ids_number_verdict_rows_in_order() -> None:
    records = _records(result_to_jsonl(_result(
        ("a", True, "", False, False),
        ("b", False, "wrong", False, False),
    )))
    assert [record["id"] for record in records] == ["1", "2"]


def test_phase_entries_are_skipped() -> None:
    result = {
        "verification_log": [
            {"phase": "bypassed", "reason": "strictness=0.0"},
            {"claim": "kept", "is_valid": True, "critique": ""},
            {"skipped_by_budget": True, "phase": "correction"},
        ],
    }
    records = _records(result_to_jsonl(result))
    assert [record["claim"] for record in records] == ["kept"]


def test_full_session_payloads_are_accepted(tmp_path) -> None:
    path = tmp_path / "session.json"
    save_session(
        path, prompt="p", config={"model": "m"},
        result=_result(("x", False, "wrong", True, False)),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    records = _records(result_to_jsonl(payload))
    assert len(records) == 1
    assert records[0]["verdict"] == "flagged"


def test_cli_prints_jsonl_to_stdout(tmp_path, capsys) -> None:
    session_path = str(tmp_path / "s.json")
    save_session(
        session_path, prompt="p", config={},
        result=_result(("k", True, "", False, False)),
    )

    args = _build_parser().parse_args(["export-jsonl", session_path])
    assert _cmd_export_jsonl(args) == 0
    records = _records(capsys.readouterr().out)
    assert len(records) == 1
    assert records[0]["claim"] == "k"


def test_cli_writes_output_file(tmp_path, capsys) -> None:
    session_path = tmp_path / "s.json"
    output_path = tmp_path / "nested" / "verdicts.jsonl"
    save_session(
        session_path, prompt="p", config={},
        result=_result(("f", False, "bad", False, False)),
    )

    args = _build_parser().parse_args(
        ["export-jsonl", str(session_path), "--output", str(output_path)]
    )
    assert _cmd_export_jsonl(args) == 0
    records = _records(output_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["verdict"] == "flagged"
    assert "JSONL report written" in capsys.readouterr().out


def test_cli_rejects_unreadable_sessions(tmp_path, capsys) -> None:
    args = _build_parser().parse_args(["export-jsonl", str(tmp_path / "missing.json")])
    assert _cmd_export_jsonl(args) == 2
    assert "export-jsonl:" in capsys.readouterr().err


def test_subcommand_is_registered() -> None:
    args = _build_parser().parse_args(["export-jsonl", "session.json"])
    assert args.session == "session.json"
    assert args.output is None
