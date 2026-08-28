"""Tests for exporting saved sessions as per-claim CSV verdicts."""

from __future__ import annotations

import csv
import io
import json

from self_correct.cli import _build_parser, _cmd_export_csv
from self_correct.csvreport import result_to_csv
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


def _rows(text: str):
    return list(csv.DictReader(io.StringIO(text)))


def test_each_verdict_becomes_a_row() -> None:
    text = result_to_csv(_result(
        ("sky is blue", True, "", False, False),
        ("moon is cheese", False, "Moon is rock.", True, False),
    ))
    rows = _rows(text)
    assert [row["claim"] for row in rows] == ["sky is blue", "moon is cheese"]
    assert rows[0]["verdict"] == "verified"
    assert rows[1]["verdict"] == "flagged"
    assert rows[1]["evidence_used"] == "true"


def test_flagged_rows_carry_severity_and_critique() -> None:
    rows = _rows(result_to_csv(_result(
        ("x", False, "Statement is false.", False, False),
    )))
    assert rows[0]["severity"] == "critical"
    assert rows[0]["critique"] == "Statement is false."


def test_verified_rows_have_empty_severity() -> None:
    rows = _rows(result_to_csv(_result(
        ("y", True, "", True, True),
    )))
    assert rows[0]["severity"] == ""
    assert rows[0]["cached"] == "true"


def test_ids_number_verdict_rows_in_order() -> None:
    rows = _rows(result_to_csv(_result(
        ("a", True, "", False, False),
        ("b", False, "wrong", False, False),
    )))
    assert [row["id"] for row in rows] == ["1", "2"]


def test_critiques_with_commas_are_quoted() -> None:
    rows = _rows(result_to_csv(_result(
        ("x", False, "misleading, inaccurate and unverifiable", False, False),
    )))
    assert rows[0]["critique"] == "misleading, inaccurate and unverifiable"


def test_phase_entries_are_skipped() -> None:
    result = {
        "verification_log": [
            {"phase": "bypassed", "reason": "strictness=0.0"},
            {"claim": "kept", "is_valid": True, "critique": ""},
            {"skipped_by_budget": True, "phase": "correction"},
        ],
    }
    rows = _rows(result_to_csv(result))
    assert [row["claim"] for row in rows] == ["kept"]


def test_full_session_payloads_are_accepted(tmp_path) -> None:
    path = tmp_path / "session.json"
    save_session(
        path, prompt="p", config={"model": "m"},
        result=_result(("x", False, "wrong", True, False)),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    rows = _rows(result_to_csv(payload))
    assert len(rows) == 1
    assert rows[0]["verdict"] == "flagged"


def test_cli_prints_csv_to_stdout(tmp_path, capsys) -> None:
    session_path = str(tmp_path / "s.json")
    save_session(
        session_path, prompt="p", config={},
        result=_result(("k", True, "", False, False)),
    )

    args = _build_parser().parse_args(["export-csv", session_path])
    assert _cmd_export_csv(args) == 0
    rows = _rows(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["claim"] == "k"


def test_cli_writes_output_file(tmp_path, capsys) -> None:
    session_path = tmp_path / "s.json"
    output_path = tmp_path / "nested" / "verdicts.csv"
    save_session(
        session_path, prompt="p", config={},
        result=_result(("f", False, "bad", False, False)),
    )

    args = _build_parser().parse_args(
        ["export-csv", str(session_path), "--output", str(output_path)]
    )
    assert _cmd_export_csv(args) == 0
    rows = _rows(output_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["verdict"] == "flagged"
    assert "CSV report written" in capsys.readouterr().out


def test_cli_rejects_unreadable_sessions(tmp_path, capsys) -> None:
    args = _build_parser().parse_args(["export-csv", str(tmp_path / "missing.json")])
    assert _cmd_export_csv(args) == 2
    assert "export-csv:" in capsys.readouterr().err


def test_subcommand_is_registered() -> None:
    args = _build_parser().parse_args(["export-csv", "session.json"])
    assert args.session == "session.json"
    assert args.output is None
