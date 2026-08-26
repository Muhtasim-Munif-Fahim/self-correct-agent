"""Tests for aggregate analytics across saved session files."""

from __future__ import annotations

import json
import os

from self_correct import cli, sessions


def _write_session(tmp_path, name, verdicts):
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt=f"prompt for {name}",
        config={"model": "gpt-test"},
        result={
            "content": "answer",
            "verification_log": [
                {"claim": claim, "is_valid": valid, "critique": critique}
                for claim, valid, critique in verdicts
            ],
        },
    )
    return path


def test_aggregate_totals_and_flag_rate(tmp_path) -> None:
    _write_session(tmp_path, "a.json", [("x", True, ""), ("y", True, ""), ("z", False, "")])
    _write_session(tmp_path, "b.json", [("w", False, "")])

    aggregate = sessions.aggregate_sessions([tmp_path])
    totals = aggregate["totals"]

    assert totals["sessions"] == 2
    assert totals["claims"] == 4
    assert totals["verified"] == 2
    assert totals["flagged"] == 2
    assert totals["flag_rate"] == 0.5
    assert aggregate["invalid"] == []


def test_severity_mix_follows_the_default_taxonomy(tmp_path) -> None:
    _write_session(tmp_path, "a.json", [
        ("x", False, "Statement is false."),
        ("y", False, "Unverifiable without sources."),
        ("z", False, "Wording is imprecise."),
        ("v", True, ""),
    ])

    aggregate = sessions.aggregate_sessions([tmp_path / "a.json"])
    assert aggregate["totals"]["severities"] == {"critical": 1, "major": 1, "minor": 1}


def test_directories_expand_to_json_files_without_duplicates(tmp_path) -> None:
    _write_session(tmp_path, "a.json", [])
    _write_session(tmp_path, "b.json", [])
    (tmp_path / "notes.txt").write_text("not a session", encoding="utf-8")

    aggregate = sessions.aggregate_sessions(
        [tmp_path, tmp_path / "a.json", tmp_path / "a.json"]
    )
    assert aggregate["totals"]["sessions"] == 2
    assert {row["file"] for row in aggregate["sessions"]} == {
        str(tmp_path / "a.json"),
        str(tmp_path / "b.json"),
    }


def test_invalid_files_are_reported_instead_of_aborting(tmp_path) -> None:
    good = _write_session(tmp_path, "good.json", [("x", True, "")])
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    aggregate = sessions.aggregate_sessions([broken, good])

    assert aggregate["totals"]["sessions"] == 1
    assert aggregate["totals"]["invalid_files"] == 1
    assert aggregate["invalid"][0]["file"] == str(broken)
    assert "not valid JSON" in aggregate["invalid"][0]["error"]


def test_trend_rows_are_sorted_oldest_first(tmp_path) -> None:
    newer = _write_session(tmp_path, "newer.json", [("x", True, "")])
    older = _write_session(tmp_path, "older.json", [("y", True, "")])
    os.utime(newer, (2_000_000_000, 2_000_000_000))
    os.utime(older, (1_000_000_000, 1_000_000_000))

    rows = sessions.aggregate_sessions([tmp_path])["sessions"]
    assert [row["file"] for row in rows] == [str(older), str(newer)]


def test_session_without_verdicts_scores_zero_flag_rate(tmp_path) -> None:
    _write_session(tmp_path, "empty.json", [])

    aggregate = sessions.aggregate_sessions([tmp_path / "empty.json"])
    totals = aggregate["totals"]

    assert totals["claims"] == 0
    assert totals["flag_rate"] == 0.0


def test_cli_prints_a_trend_table_with_totals(tmp_path, capsys) -> None:
    _write_session(tmp_path, "a.json", [("x", True, ""), ("z", False, "false claim")])

    exit_code = cli.main(["sessions-stats", str(tmp_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "TOTAL" in output
    assert "1 saved session(s)" in output
    assert str(tmp_path / "a.json") in output


def test_cli_json_mode_reports_machine_readable_totals(tmp_path, capsys) -> None:
    _write_session(tmp_path, "a.json", [("x", True, "")])

    exit_code = cli.main(["sessions-stats", "--json", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["totals"]["claims"] == 1
    assert payload["totals"]["verified"] == 1


def test_cli_fails_when_no_session_is_valid(tmp_path, capsys) -> None:
    (tmp_path / "broken.json").write_text("[]", encoding="utf-8")

    exit_code = cli.main(["sessions-stats", str(tmp_path)])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "No valid session files found." in err