"""Tests for the batch dry-run plan mode."""

from __future__ import annotations

import json

from self_correct.cli import _build_parser, _plan_batch_items, cmd_batch


def _write_input(tmp_path, items):
    path = tmp_path / "in.jsonl"
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in items),
        encoding="utf-8",
    )
    return path


def test_parser_accepts_dry_run() -> None:
    args = _build_parser().parse_args(["batch", "--input", "in.jsonl", "--dry-run"])
    assert args.dry_run is True
    args = _build_parser().parse_args(["batch", "--input", "in.jsonl"])
    assert args.dry_run is False


def test_plan_marks_promptless_items_as_skipped() -> None:
    items = [
        {"id": "a", "prompt": "one"},
        {"id": "b", "prompt": ""},
        {"id": "c", "prompt": "three"},
    ]
    plan = _plan_batch_items(items, {})
    assert [entry["action"] for entry in plan] == ["work", "skipped", "work"]


def test_plan_marks_completed_prior_records_as_resumed() -> None:
    items = [{"id": "a", "prompt": "one"}, {"id": "b", "prompt": "two"}]
    prior = {
        "a": {"id": "a", "content": "done"},
        "b": {"id": "b", "error": "RuntimeError: x"},
    }
    plan = _plan_batch_items(items, prior)
    assert plan[0]["action"] == "resumed"
    assert plan[0]["record"]["content"] == "done"
    assert plan[1]["action"] == "work"


def test_dry_run_prints_plan_without_writing_output(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = _write_input(tmp_path, [
        {"id": "doc-1", "prompt": "one"},
        {"id": "doc-2", "prompt": ""},
    ])
    output_path = tmp_path / "out.jsonl"

    called = {"generate": 0}

    class FakeHallucinator:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            called["generate"] += 1
            raise AssertionError("dry run must not call the LLM")

    monkeypatch.setattr("self_correct.cli.AntiHallucinator", FakeHallucinator)

    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path), "--output", str(output_path),
            "--dry-run",
        ]
    )
    assert cmd_batch(args) == 0

    out = capsys.readouterr().out
    assert "DRY RUN - no API calls will be made" in out
    assert "[1/2] process doc-1" in out
    assert "[2/2] skip    doc-2" in out
    assert called["generate"] == 0
    assert not output_path.exists()


def test_dry_run_reports_resumed_and_skipped_counts(
    tmp_path, capsys
) -> None:
    input_path = _write_input(tmp_path, [
        {"id": "done", "prompt": "x"},
        {"id": "todo", "prompt": "y"},
        {"id": "empty", "prompt": ""},
    ])
    prior_path = tmp_path / "prior.jsonl"
    prior_path.write_text(
        json.dumps({"id": "done", "content": "verified"}) + "\n",
        encoding="utf-8",
    )

    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path),
            "--resume-from", str(prior_path), "--dry-run",
        ]
    )
    assert cmd_batch(args) == 0

    out = capsys.readouterr().out
    assert "3 item(s): 1 to process, 1 resumed, 1 skipped" in out
    assert "[1/3] resume  done" in out


def test_dry_run_does_not_require_an_api_key(
    tmp_path, monkeypatch, capsys
) -> None:
    import sys
    from unittest.mock import MagicMock

    input_path = _write_input(tmp_path, [{"id": "only", "prompt": "p"}])

    fake_module = MagicMock()

    class FailOnOpenAI:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry run must not construct the OpenAI client")

    fake_module.OpenAI = FailOnOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    args = _build_parser().parse_args(
        ["batch", "--input", str(input_path), "--dry-run"]
    )
    assert cmd_batch(args) == 0
    assert "[1/1] process only" in capsys.readouterr().out
