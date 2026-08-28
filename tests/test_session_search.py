"""Tests for searching saved sessions by claim content and verdict."""

from __future__ import annotations

import json

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


def test_claim_query_matches_substrings_case_insensitively(tmp_path) -> None:
    _write_session(tmp_path, "a.json", [
        ("Paris is the capital of France.", True, ""),
        ("The moon is made of cheese.", False, "Contradicts sources."),
    ])
    _write_session(tmp_path, "b.json", [("Paris has many museums.", True, "")])

    result = sessions.search_sessions([tmp_path], claim_query="paris")

    assert result["match_count"] == 2
    assert result["scanned"] == 2
    claims = {match["claim"] for match in result["matches"]}
    assert claims == {"Paris is the capital of France.", "Paris has many museums."}


def test_verdict_filter_keeps_only_requested_outcomes(tmp_path) -> None:
    _write_session(tmp_path, "a.json", [
        ("good", True, ""),
        ("bad", False, "not true"),
    ])

    flagged = sessions.search_sessions([tmp_path], verdict="flagged")
    assert [match["claim"] for match in flagged["matches"]] == ["bad"]

    verified = sessions.search_sessions([tmp_path], verdict="verified")
    assert [match["claim"] for match in verified["matches"]] == ["good"]


def test_critique_query_matches_critique_text(tmp_path) -> None:
    _write_session(tmp_path, "a.json", [
        ("x", False, "The claim contradicts official records."),
        ("y", False, "Wording is imprecise."),
    ])

    result = sessions.search_sessions([tmp_path], critique_query="contradicts")

    assert result["match_count"] == 1
    assert result["matches"][0]["critique"] == "The claim contradicts official records."


def test_filters_combine_with_and(tmp_path) -> None:
    _write_session(tmp_path, "a.json", [
        ("Paris is big.", True, ""),
        ("Paris is a city.", False, "Unverifiable."),
    ])

    result = sessions.search_sessions(
        [tmp_path], claim_query="paris", verdict="flagged"
    )

    assert [match["claim"] for match in result["matches"]] == ["Paris is a city."]


def test_no_matches_returns_empty_list(tmp_path) -> None:
    _write_session(tmp_path, "a.json", [("London exists.", True, "")])

    result = sessions.search_sessions([tmp_path], claim_query="nowhere")

    assert result["match_count"] == 0
    assert result["matches"] == []
    assert result["scanned"] == 1


def test_invalid_files_are_skipped_and_reported(tmp_path) -> None:
    good = _write_session(tmp_path, "good.json", [("hits here", True, "")])
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    result = sessions.search_sessions([broken, good], claim_query="hits")

    assert result["match_count"] == 1
    assert result["scanned"] == 1
    assert result["invalid"][0]["file"] == str(broken)
    assert "not valid JSON" in result["invalid"][0]["error"]


def test_entries_without_verdicts_are_ignored(tmp_path) -> None:
    path = tmp_path / "a.json"
    sessions.save_session(
        path,
        prompt="p",
        config={},
        result={
            "verification_log": [
                {"phase": "bypassed", "reason": "strictness=0.0"},
                {"skipped_by_budget": True, "phase": "correction"},
                {"claim": "kept claim", "is_valid": True, "critique": ""},
            ],
        },
    )

    result = sessions.search_sessions([path], claim_query="kept")

    assert result["match_count"] == 1
    assert result["matches"][0]["claim"] == "kept claim"


def test_cli_prints_matches_with_locations(tmp_path, capsys) -> None:
    _write_session(tmp_path, "a.json", [
        ("Paris is the capital.", False, "Unverifiable without sources."),
    ])

    exit_code = cli.main(["sessions-search", str(tmp_path), "--claim", "paris"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "1 match(es) across 1 session file(s)" in output
    assert "Paris is the capital." in output
    assert "critique: Unverifiable without sources." in output


def test_cli_json_mode_reports_matches(tmp_path, capsys) -> None:
    _write_session(tmp_path, "a.json", [("Paris is big.", True, "")])

    exit_code = cli.main(
        ["sessions-search", "--json", str(tmp_path), "--claim", "paris"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["match_count"] == 1
    assert payload["matches"][0]["verdict"] == "verified"


def test_cli_requires_at_least_one_filter(tmp_path, capsys) -> None:
    exit_code = cli.main(["sessions-search", str(tmp_path)])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "--claim" in err


def test_cli_fails_when_no_session_is_valid(tmp_path, capsys) -> None:
    (tmp_path / "broken.json").write_text("[]", encoding="utf-8")

    exit_code = cli.main(
        ["sessions-search", str(tmp_path), "--verdict", "flagged"]
    )
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "No valid session files found." in err
