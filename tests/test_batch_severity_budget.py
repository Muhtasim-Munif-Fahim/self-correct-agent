"""Tests for the per-batch severity budget gate."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from self_correct.cli import (
    _build_parser,
    _non_negative_int,
    _severity_budget_reasons,
    cmd_batch,
)


def _install_fake_pipeline(monkeypatch, responses):
    fake_module = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    class FakeHallucinator:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return SimpleNamespace(to_dict=lambda: responses.pop(0))

    monkeypatch.setattr("self_correct.cli.AntiHallucinator", FakeHallucinator)


def _write_input(tmp_path, ids):
    path = tmp_path / "in.jsonl"
    path.write_text(
        "".join(json.dumps({"id": i, "prompt": i}) + "\n" for i in ids),
        encoding="utf-8",
    )
    return path


def test_non_negative_int_accepts_zero_and_positive() -> None:
    assert _non_negative_int("0") == 0
    assert _non_negative_int("5") == 5


def test_non_negative_int_rejects_negative() -> None:
    with pytest.raises(Exception, match="non-negative"):
        _non_negative_int("-1")


def test_parser_accepts_severity_budgets() -> None:
    args = _build_parser().parse_args(
        [
            "batch", "--input", "in.jsonl",
            "--max-critical", "2", "--max-major", "3", "--max-minor", "5",
        ]
    )
    assert args.max_critical == 2
    assert args.max_major == 3
    assert args.max_minor == 5


def test_parser_defaults_severity_budgets_to_unlimited() -> None:
    args = _build_parser().parse_args(["batch", "--input", "in.jsonl"])
    assert args.max_critical is None
    assert args.max_major is None
    assert args.max_minor is None


def test_parser_rejects_negative_budget() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["batch", "--input", "in.jsonl", "--max-critical", "-1"]
        )


def test_severity_budget_reasons_unlimited_when_unset() -> None:
    totals = {"critical": 9, "major": 9, "minor": 9}
    args = SimpleNamespace(max_critical=None, max_major=None, max_minor=None)
    assert _severity_budget_reasons(totals, args) == []


def test_severity_budget_reasons_within_budget() -> None:
    totals = {"critical": 1, "major": 2, "minor": 3}
    args = SimpleNamespace(max_critical=1, max_major=2, max_minor=5)
    assert _severity_budget_reasons(totals, args) == []


def test_severity_budget_reasons_exceeded() -> None:
    totals = {"critical": 3, "major": 1, "minor": 0}
    args = SimpleNamespace(max_critical=2, max_major=None, max_minor=None)
    reasons = _severity_budget_reasons(totals, args)
    assert reasons == ["critical claims 3 exceed 2"]


def test_severity_budget_zero_fails_when_present() -> None:
    totals = {"critical": 0, "major": 2, "minor": 0}
    args = SimpleNamespace(max_critical=None, max_major=0, max_minor=None)
    reasons = _severity_budget_reasons(totals, args)
    assert reasons == ["major claims 2 exceed 0"]


def test_batch_fails_when_severity_budget_exceeded(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = _write_input(tmp_path, ["doc-1"])
    _install_fake_pipeline(
        monkeypatch,
        [
            {
                "id": "doc-1",
                "content": "text",
                "hallucinations_caught": ["bad"],
                "verification_log": [
                    {"claim": "c", "is_valid": False, "critique": "incorrect"}
                ],
                "severity_summary": {"critical": 3, "major": 0, "minor": 0},
            }
        ],
    )
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path),
            "--max-critical", "2",
        ]
    )
    assert cmd_batch(args) == 1
    err = capsys.readouterr().err
    assert "critical claims 3 exceed 2" in err
    assert "Batch severity budget exceeded" in err


def test_batch_passes_within_severity_budget(tmp_path, monkeypatch) -> None:
    input_path = _write_input(tmp_path, ["doc-1"])
    _install_fake_pipeline(
        monkeypatch,
        [
            {
                "id": "doc-1",
                "content": "text",
                "hallucinations_caught": [],
                "verification_log": [
                    {"claim": "c", "is_valid": False, "critique": "incorrect"}
                ],
                "severity_summary": {"critical": 1, "major": 0, "minor": 0},
            }
        ],
    )
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path),
            "--max-critical", "2",
        ]
    )
    assert cmd_batch(args) == 0


def test_severity_budget_is_additive_with_fail_on_hallucination(
    tmp_path, monkeypatch
) -> None:
    input_path = _write_input(tmp_path, ["doc-1"])
    _install_fake_pipeline(
        monkeypatch,
        [
            {
                "id": "doc-1",
                "content": "text",
                "hallucinations_caught": ["flagged"],
                "verification_log": [
                    {"claim": "c", "is_valid": False, "critique": "incorrect"}
                ],
                "severity_summary": {"critical": 5, "major": 0, "minor": 0},
            }
        ],
    )
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path),
            "--fail-on-hallucination", "--max-critical", "2",
        ]
    )
    assert cmd_batch(args) == 1


def test_severity_budget_aggregates_across_items(tmp_path, monkeypatch) -> None:
    input_path = _write_input(tmp_path, ["a", "b"])
    _install_fake_pipeline(
        monkeypatch,
        [
            {
                "id": "a",
                "content": "t",
                "hallucinations_caught": [],
                "verification_log": [
                    {"claim": "c", "is_valid": False, "critique": "incorrect"}
                ],
                "severity_summary": {"critical": 2, "major": 0, "minor": 0},
            },
            {
                "id": "b",
                "content": "t",
                "hallucinations_caught": [],
                "verification_log": [
                    {"claim": "c", "is_valid": False, "critique": "incorrect"}
                ],
                "severity_summary": {"critical": 2, "major": 0, "minor": 0},
            },
        ],
    )
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path),
            "--max-critical", "3",
        ]
    )
    assert cmd_batch(args) == 1
