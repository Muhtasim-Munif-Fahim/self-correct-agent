"""Tests for exporting saved sessions as per-claim Markdown verdicts."""

from __future__ import annotations

import json
from pathlib import Path

from self_correct.cli import _build_parser, _cmd_export_markdown
from self_correct.mdreport import MD_COLUMNS, result_to_markdown, verdict_rows
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


def test_each_verdict_becomes_a_row() -> None:
    rows = list(verdict_rows(_result(
        ("sky is blue", True, "", False, False),
        ("moon is cheese", False, "Moon is rock.", True, False),
    )))
    assert [row["claim"] for row in rows] == ["sky is blue", "moon is cheese"]
    assert rows[0]["verdict"] == "verified"
    assert rows[1]["verdict"] == "flagged"
    assert rows[1]["evidence_used"] == "true"


def test_flagged_rows_carry_severity_and_critique() -> None:
    rows = list(verdict_rows(_result(
        ("x", False, "Statement is false.", False, False),
    )))
    assert rows[0]["severity"] == "critical"
    assert rows[0]["critique"] == "Statement is false."


def test_verified_rows_have_empty_severity() -> None:
    rows = list(verdict_rows(_result(
        ("y", True, "", True, True),
    )))
    assert rows[0]["severity"] == ""
    assert rows[0]["cached"] == "true"


def test_ids_number_verdict_rows_in_order() -> None:
    rows = list(verdict_rows(_result(
        ("a", True, "", False, False),
        ("b", False, "wrong", False, False),
    )))
    assert [row["id"] for row in rows] == ["1", "2"]


def test_phase_entries_are_skipped() -> None:
    result = {
        "verification_log": [
            {"phase": "bypassed", "reason": "strictness=0.0"},
            {"claim": "kept", "is_valid": True, "critique": ""},
            {"skipped_by_budget": True, "phase": "correction"},
        ],
    }
    rows = list(verdict_rows(result))
    assert [row["claim"] for row in rows] == ["kept"]


def test_full_session_payloads_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    save_session(
        path, prompt="p", config={"model": "m"},
        result=_result(("x", False, "wrong", True, False)),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    rows = list(verdict_rows(payload))
    assert len(rows) == 1
    assert rows[0]["verdict"] == "flagged"


def test_result_to_markdown_has_header_and_separator() -> None:
    text = result_to_markdown(_result(
        ("sky is blue", True, "", False, False),
    ))
    lines = text.strip().split("\n")
    assert lines[0] == "| " + " | ".join(MD_COLUMNS) + " |"
    assert lines[1] == "| " + " | ".join("---" for _ in MD_COLUMNS) + " |"


def test_result_to_markdown_renders_one_row_per_claim() -> None:
    text = result_to_markdown(_result(
        ("sky is blue", True, "", False, False),
        ("moon is cheese", False, "is false", False, False),
    ))
    lines = text.strip().split("\n")
    assert len(lines) == 4
    assert lines[2].startswith("| 1 | sky is blue | verified")
    assert lines[3].startswith("| 2 | moon is cheese | flagged | critical")


def test_pipe_characters_in_critique_are_escaped() -> None:
    text = result_to_markdown(_result(
        ("x", False, "false | misleading", False, False),
    ))
    lines = text.strip().split("\n")
    assert "false \\| misleading" in lines[2]


def test_newlines_in_claim_are_escaped() -> None:
    text = result_to_markdown(_result(
        ("line one\ntwo", True, "", False, False),
    ))
    lines = text.strip().split("\n")
    assert len(lines) == 3
    assert "line one two" in lines[2]


def test_empty_log_yields_header_only() -> None:
    text = result_to_markdown({"verification_log": []})
    assert text.strip() == "| " + " | ".join(MD_COLUMNS) + " |\n| " + " | ".join("---" for _ in MD_COLUMNS) + " |"


def test_cli_prints_markdown_to_stdout(tmp_path: Path, capsys) -> None:
    session_path = tmp_path / "s.json"
    save_session(
        session_path, prompt="p", config={},
        result=_result(("k", True, "", False, False)),
    )

    args = _build_parser().parse_args(["export-markdown", str(session_path)])
    assert _cmd_export_markdown(args) == 0
    text = capsys.readouterr().out
    assert "| claim |" in text
    assert "k | verified" in text


def test_cli_writes_output_file(tmp_path: Path) -> None:
    session_path = tmp_path / "s.json"
    output_path = tmp_path / "nested" / "verdicts.md"
    save_session(
        session_path, prompt="p", config={},
        result=_result(("f", False, "bad", False, False)),
    )

    args = _build_parser().parse_args(
        ["export-markdown", str(session_path), "--output", str(output_path)]
    )
    assert _cmd_export_markdown(args) == 0
    text = output_path.read_text(encoding="utf-8")
    assert "f | flagged | minor |" in text
    assert output_path.exists()


def test_cli_rejects_unreadable_sessions(tmp_path: Path) -> None:
    args = _build_parser().parse_args(["export-markdown", str(tmp_path / "missing.json")])
    assert _cmd_export_markdown(args) == 2


def test_subcommand_is_registered() -> None:
    args = _build_parser().parse_args(["export-markdown", "session.json"])
    assert args.session == "session.json"
    assert args.output is None
