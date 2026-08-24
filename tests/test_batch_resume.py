"""Tests for resuming interrupted batch runs from prior output."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from self_correct.cli import (
    _build_parser,
    _is_completed,
    _load_prior_results,
    cmd_batch,
)


def _install_fake_pipeline(monkeypatch, calls, responses):
    fake_module = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    class FakeHallucinator:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(to_dict=lambda: responses.pop(0))

    monkeypatch.setattr("self_correct.cli.AntiHallucinator", FakeHallucinator)


def test_prior_output_accepts_jsonl_and_array_formats(tmp_path) -> None:
    jsonl = tmp_path / "out.jsonl"
    jsonl.write_text(
        json.dumps({"id": 1, "content": "a"}) + "\n"
        + json.dumps({"id": "2", "error": "RuntimeError: x"}) + "\n",
        encoding="utf-8",
    )
    assert _load_prior_results(str(jsonl)) == {
        "1": {"id": 1, "content": "a"},
        "2": {"id": "2", "error": "RuntimeError: x"},
    }

    array = tmp_path / "out.json"
    array.write_text(json.dumps([{"id": "only", "content": "b"}]), encoding="utf-8")
    assert _load_prior_results(str(array)) == {"only": {"id": "only", "content": "b"}}


def test_malformed_prior_lines_are_reported(tmp_path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": 1}) + "\n{oops\n", encoding="utf-8")
    try:
        _load_prior_results(str(bad))
    except ValueError as exc:
        assert "line 2" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_completed_records_exclude_errors() -> None:
    assert _is_completed({"id": 1, "content": "text"}) is True
    assert _is_completed({"id": 2, "error": "RuntimeError: down"}) is False
    assert _is_completed(None) is False


def test_parser_accepts_resume_from() -> None:
    args = _build_parser().parse_args(
        ["batch", "--input", "in.jsonl", "--resume-from", "prior.jsonl"]
    )
    assert args.resume_from == "prior.jsonl"


def test_resume_reuses_done_items_and_retries_failed_ones(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        json.dumps({"id": "doc-1", "prompt": "one"}) + "\n"
        + json.dumps({"id": "doc-2", "prompt": "two"}) + "\n",
        encoding="utf-8",
    )
    prior_path = tmp_path / "prior.jsonl"
    prior_path.write_text(
        json.dumps(
            {
                "id": "doc-1",
                "prompt": "one",
                "content": "original verified text",
                "hallucinations_caught": [],
            }
        )
        + "\n"
        + json.dumps({"id": "doc-2", "prompt": "two", "error": "ConnectionError: x"})
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "final.jsonl"

    calls: list = []
    _install_fake_pipeline(
        monkeypatch,
        calls,
        [
            {
                "id": "doc-2",
                "prompt": "two",
                "content": "retried text",
                "hallucinations_caught": [],
            }
        ],
    )

    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path), "--output", str(output_path),
            "--resume-from", str(prior_path), "--format", "jsonl",
        ]
    )
    cmd_batch(args)
    err = capsys.readouterr().err
    assert "'doc-1' already done" in err
    assert "(1 resumed)" in err

    # Only the failed item was regenerated.
    assert len(calls) == 1
    assert calls[0]["prompt"] == "two"

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in records}
    assert by_id["doc-1"]["content"] == "original verified text"
    assert by_id["doc-2"]["content"] == "retried text"
    assert "error" not in by_id["doc-2"]


def test_unreadable_resume_file_exits_with_status_two(tmp_path, capsys) -> None:
    args = _build_parser().parse_args(
        ["batch", "--input", "in.jsonl", "--resume-from", str(tmp_path / "nope.jsonl")]
    )
    assert cmd_batch(args) == 2
    assert "batch:" in capsys.readouterr().err
