"""Tests for exporting saved sessions as JUnit XML."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from self_correct.cli import _build_parser, _cmd_export_junit
from self_correct.junit import result_to_junit_xml
from self_correct.sessions import save_session


def _result(*claims, elapsed=None):
    result = {
        "content": "answer",
        "verification_log": [
            {"claim": claim, "is_valid": is_valid, "critique": critique}
            for claim, is_valid, critique in claims
        ],
    }
    if elapsed is not None:
        result["elapsed_seconds"] = elapsed
    return result


def test_flagged_claims_become_failures() -> None:
    xml_text = result_to_junit_xml(_result(
        ("sky is blue", True, ""),
        ("moon is cheese", False, "Moon is rock."),
    ))

    suite = ET.fromstring(xml_text)
    assert suite.tag == "testsuite"
    assert suite.get("tests") == "2"
    assert suite.get("failures") == "1"

    cases = suite.findall("testcase")
    assert [case.get("name") for case in cases] == ["sky is blue", "moon is cheese"]
    failures = cases[1].findall("failure")
    assert len(failures) == 1
    assert failures[0].get("message") == "Moon is rock."
    assert cases[0].findall("failure") == []


def test_phase_entries_are_skipped() -> None:
    result = {
        "verification_log": [
            {"phase": "bypassed", "reason": "strictness=0.0"},
            {"claim": "kept", "is_valid": True, "critique": ""},
            {"skipped_by_budget": True, "phase": "correction"},
        ],
    }
    suite = ET.fromstring(result_to_junit_xml(result))
    assert suite.get("tests") == "1"
    assert suite.get("failures") == "0"


def test_elapsed_seconds_is_reported() -> None:
    suite = ET.fromstring(
        result_to_junit_xml(_result(("c", True, ""), elapsed=12.3456))
    )
    assert suite.get("time") == "12.346"

    no_time = ET.fromstring(result_to_junit_xml(_result(("c", True, ""))))
    assert no_time.get("time") is None


def test_special_characters_do_not_break_the_xml() -> None:
    xml_text = result_to_junit_xml(_result(
        ("a < b & c", False, 'contradicts "recorded" history'),
    ))
    suite = ET.fromstring(xml_text)
    failure = suite.find("testcase/failure")
    assert failure.get("message") == 'contradicts "recorded" history'


def test_full_session_payloads_are_accepted(tmp_path) -> None:
    path = tmp_path / "session.json"
    save_session(path, prompt="p", config={"model": "m"}, result=_result(("x", False, "wrong")))
    payload = json.loads(path.read_text(encoding="utf-8"))

    suite = ET.fromstring(result_to_junit_xml(payload))
    assert suite.get("failures") == "1"


def test_cli_prints_xml_to_stdout(tmp_path, capsys) -> None:
    session_path = str(tmp_path / "s.json")
    save_session(session_path, prompt="p", config={}, result=_result(("k", True, "")))

    args = _build_parser().parse_args(["export-junit", session_path])
    assert _cmd_export_junit(args) == 0
    suite = ET.fromstring(capsys.readouterr().out)
    assert suite.get("tests") == "1"


def test_cli_writes_output_file(tmp_path, capsys) -> None:
    session_path = tmp_path / "s.json"
    output_path = tmp_path / "nested" / "report.xml"
    save_session(session_path, prompt="p", config={}, result=_result(("f", False, "bad")))

    args = _build_parser().parse_args(
        ["export-junit", str(session_path), "--output", str(output_path)]
    )
    assert _cmd_export_junit(args) == 0
    suite = ET.fromstring(output_path.read_text(encoding="utf-8"))
    assert suite.get("failures") == "1"
    assert "JUnit report written" in capsys.readouterr().out


def test_cli_rejects_unreadable_sessions(tmp_path, capsys) -> None:
    args = _build_parser().parse_args(["export-junit", str(tmp_path / "missing.json")])
    assert _cmd_export_junit(args) == 2
    assert "export-junit:" in capsys.readouterr().err


def test_subcommand_is_registered() -> None:
    args = _build_parser().parse_args(["export-junit", "session.json"])
    assert args.session == "session.json"
    assert args.output is None
