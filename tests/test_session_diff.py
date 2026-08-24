"""Tests for comparing claim verdicts across saved sessions."""

from __future__ import annotations

import json

import pytest

from self_correct.cli import _build_parser, _cmd_session_diff
from self_correct.sessions import diff_sessions, save_session


def _result(*claims):
    return {
        "content": "answer",
        "verification_log": [
            {"claim": claim, "is_valid": is_valid} for claim, is_valid in claims
        ],
    }


def test_diff_classifies_resolved_and_regressed_claims() -> None:
    before = _result(("kept", True), ("fixed", False), ("broke", True))
    after = _result(("kept", True), ("fixed", True), ("broke", False))
    diff = diff_sessions(before, after)

    assert diff["resolved"] == ["fixed"]
    assert diff["regressed"] == ["broke"]
    assert diff["unchanged_verified"] == 1
    assert diff["unchanged_flagged"] == 0
    assert diff["baseline_claims"] == 3
    assert diff["current_claims"] == 3


def test_diff_tracks_added_and_removed_claims() -> None:
    before = _result(("old", True))
    after = _result(("new", False))
    diff = diff_sessions(before, after)

    assert diff["added"] == ["new"]
    assert diff["removed"] == ["old"]
    assert diff["resolved"] == []
    assert diff["regressed"] == []


def test_claim_matching_normalizes_case_and_whitespace() -> None:
    before = _result(("  The Sky IS Blue ", True))
    after = _result(("the sky is blue", False))
    diff = diff_sessions(before, after)
    assert diff["regressed"] == ["the sky is blue"]


def test_phase_entries_are_ignored() -> None:
    before = {"verification_log": [{"phase": "bypassed", "reason": "x"}]}
    after = _result(("claim", True))
    diff = diff_sessions(before, after)
    assert diff["baseline_claims"] == 0
    assert diff["added"] == ["claim"]


def test_diff_accepts_full_session_payloads(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_session(first, prompt="p", config={}, result=_result(("a", False)))
    save_session(second, prompt="p", config={}, result=_result(("a", True)))

    from self_correct.sessions import load_session

    diff = diff_sessions(load_session(first), load_session(second))
    assert diff["resolved"] == ["a"]


def _write_session(path, result) -> str:
    save_session(path, prompt="prompt", config={"model": "m"}, result=result)
    return str(path)


def test_cli_reports_and_gates_on_regression(tmp_path, capsys) -> None:
    baseline_path = _write_session(tmp_path / "a.json", _result(("x", True)))
    current_path = _write_session(tmp_path / "b.json", _result(("x", False)))

    args = _build_parser().parse_args(
        ["session-diff", baseline_path, current_path, "--fail-on-regression"]
    )
    assert _cmd_session_diff(args) == 1
    out = capsys.readouterr().out
    assert "Regressed (verified -> flagged): 1" in out

    clean_args = _build_parser().parse_args(
        ["session-diff", baseline_path, baseline_path]
    )
    assert _cmd_session_diff(clean_args) == 0


def test_cli_json_output_is_machine_readable(tmp_path, capsys) -> None:
    baseline_path = _write_session(tmp_path / "a.json", _result(("k", True), ("f", False)))
    current_path = _write_session(tmp_path / "b.json", _result(("k", True), ("f", True)))

    args = _build_parser().parse_args(
        ["session-diff", baseline_path, current_path, "--json"]
    )
    assert _cmd_session_diff(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolved"] == ["f"]


def test_cli_rejects_unreadable_sessions(tmp_path, capsys) -> None:
    args = _build_parser().parse_args(
        ["session-diff", str(tmp_path / "missing.json"), str(tmp_path / "also.json")]
    )
    assert _cmd_session_diff(args) == 2
    assert "session-diff:" in capsys.readouterr().err


def test_subcommand_is_registered() -> None:
    args = _build_parser().parse_args(["session-diff", "a.json", "b.json"])
    assert args.baseline == "a.json"
    assert args.current == "b.json"
    assert args.fail_on_regression is False
