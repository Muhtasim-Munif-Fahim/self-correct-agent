"""Tests for concurrent batch item processing with --jobs."""

from __future__ import annotations

import json
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from self_correct.cli import _build_parser, cmd_batch


def _install_parallel_pipeline(monkeypatch):
    """Replace the batch pipeline with a fake tracking peak concurrency."""

    fake_module = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    class FakeHallucinator:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            try:
                prompt = kwargs["prompt"]
                return SimpleNamespace(
                    to_dict=lambda: {
                        "id": prompt,
                        "prompt": prompt,
                        "content": "done:" + prompt,
                        "hallucinations_caught": [],
                    }
                )
            finally:
                with lock:
                    state["active"] -= 1

    monkeypatch.setattr("self_correct.cli.AntiHallucinator", FakeHallucinator)
    return state


def _write_input(tmp_path, ids):
    path = tmp_path / "in.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": item_id, "prompt": item_id}) + "\n"
            for item_id in ids
        ),
        encoding="utf-8",
    )
    return path


def test_parser_accepts_jobs_and_defaults_to_serial() -> None:
    args = _build_parser().parse_args(["batch", "--input", "in.jsonl"])
    assert args.jobs == 1
    args = _build_parser().parse_args(
        ["batch", "--input", "in.jsonl", "--jobs", "4"]
    )
    assert args.jobs == 4


def test_jobs_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["batch", "--input", "in.jsonl", "--jobs", "0"])
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["batch", "--input", "in.jsonl", "--jobs", "-2"])


def test_jobs_processes_items_concurrently_in_input_order(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = _write_input(tmp_path, ["doc-1", "doc-2", "doc-3"])
    output_path = tmp_path / "out.jsonl"

    state = _install_parallel_pipeline(monkeypatch)
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path), "--output", str(output_path),
            "--jobs", "2",
        ]
    )
    cmd_batch(args)

    assert state["max_active"] == 2

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["id"] for record in records] == ["doc-1", "doc-2", "doc-3"]
    assert [record["content"] for record in records] == [
        "done:doc-1", "done:doc-2", "done:doc-3",
    ]


def test_jobs_resume_reuses_completed_items(tmp_path, monkeypatch, capsys) -> None:
    input_path = _write_input(tmp_path, ["done-id", "pending-id"])
    prior_path = tmp_path / "prior.jsonl"
    prior_path.write_text(
        json.dumps(
            {
                "id": "done-id",
                "prompt": "done-id",
                "content": "already verified",
                "hallucinations_caught": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"

    state = _install_parallel_pipeline(monkeypatch)
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path), "--output", str(output_path),
            "--resume-from", str(prior_path), "--jobs", "2",
        ]
    )
    cmd_batch(args)
    err = capsys.readouterr().err

    assert "'done-id' already done" in err
    assert "(1 resumed)" in err
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    by_id = {record["id"]: record for record in records}
    assert by_id["done-id"]["content"] == "already verified"
    assert by_id["pending-id"]["content"] == "done:pending-id"


def test_jobs_single_item_failure_does_not_drop_other_items(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = _write_input(tmp_path, ["ok", "bad", "also-ok"])
    output_path = tmp_path / "out.jsonl"

    fake_module = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    class FakeHallucinator:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            if kwargs["prompt"] == "bad":
                raise ConnectionError("backend down")
            return SimpleNamespace(
                to_dict=lambda: {
                    "id": kwargs["prompt"],
                    "prompt": kwargs["prompt"],
                    "content": "ok:" + kwargs["prompt"],
                    "hallucinations_caught": [],
                }
            )

    monkeypatch.setattr("self_correct.cli.AntiHallucinator", FakeHallucinator)
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path), "--output", str(output_path),
            "--jobs", "2",
        ]
    )
    cmd_batch(args)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["id"] for record in records] == ["ok", "bad", "also-ok"]
    assert "error" in records[1]
    assert "backend down" in records[1]["error"]
    assert "error" not in records[0]
    assert "error" not in records[2]
