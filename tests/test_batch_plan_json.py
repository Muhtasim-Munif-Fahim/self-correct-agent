"""Tests for the machine-readable batch plan file (--plan-json)."""

from __future__ import annotations

import json

from self_correct.cli import _build_parser, _plan_batch_items, _write_plan_json, cmd_batch


def _write_input(tmp_path, items):
    path = tmp_path / "in.jsonl"
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in items),
        encoding="utf-8",
    )
    return path


def test_parser_accepts_plan_json() -> None:
    args = _build_parser().parse_args(
        ["batch", "--input", "in.jsonl", "--plan-json", "plan.json"]
    )
    assert args.plan_json == "plan.json"


def test_plan_json_defaults_to_none() -> None:
    args = _build_parser().parse_args(["batch", "--input", "in.jsonl"])
    assert args.plan_json is None


def test_write_plan_json_shape(tmp_path) -> None:
    plan = _plan_batch_items(
        [
            {"id": "a", "prompt": "one"},
            {"id": "b", "prompt": ""},
            {"id": "c", "prompt": "three"},
        ],
        {},
    )
    path = tmp_path / "plan.json"
    _write_plan_json(path, plan)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == 3
    assert payload["counts"] == {"work": 2, "resumed": 0, "skipped": 1}
    assert [item["action"] for item in payload["items"]] == ["work", "skipped", "work"]
    assert payload["items"][0]["index"] == 1
    assert payload["items"][0]["id"] == "a"
    assert payload["items"][0]["prompt"] == "one"
    assert payload["items"][0]["record"] is None


def test_write_plan_json_includes_resumed_record(tmp_path) -> None:
    prior = {"a": {"id": "a", "content": "done"}}
    plan = _plan_batch_items(
        [{"id": "a", "prompt": "one"}], prior
    )
    path = tmp_path / "plan.json"
    _write_plan_json(path, plan)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["counts"]["resumed"] == 1
    assert payload["items"][0]["record"] == {"id": "a", "content": "done"}


def test_plan_json_writes_file_without_api_calls(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = _write_input(tmp_path, [
        {"id": "doc-1", "prompt": "one"},
        {"id": "doc-2", "prompt": ""},
    ])
    plan_path = tmp_path / "plan.json"

    class FailOnOpenAI:
        def __init__(self, *args, **kwargs):
            raise AssertionError("plan-json must not construct the OpenAI client")

    import sys
    from unittest.mock import MagicMock

    fake_module = MagicMock()
    fake_module.OpenAI = FailOnOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    args = _build_parser().parse_args(
        ["batch", "--input", str(input_path), "--plan-json", str(plan_path)]
    )
    assert cmd_batch(args) == 0

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["total"] == 2
    assert payload["counts"] == {"work": 1, "resumed": 0, "skipped": 1}
    assert [item["id"] for item in payload["items"]] == ["doc-1", "doc-2"]
    assert "Batch plan written to" in capsys.readouterr().err


def test_plan_json_creates_nested_directories(tmp_path, monkeypatch) -> None:
    input_path = _write_input(tmp_path, [{"id": "only", "prompt": "p"}])
    plan_path = tmp_path / "nested" / "deep" / "plan.json"

    import sys
    from unittest.mock import MagicMock

    monkeypatch.setitem(sys.modules, "openai", MagicMock())

    args = _build_parser().parse_args(
        ["batch", "--input", str(input_path), "--plan-json", str(plan_path)]
    )
    assert cmd_batch(args) == 0
    assert plan_path.exists()


def test_plan_json_does_not_run_items(tmp_path, monkeypatch) -> None:
    input_path = _write_input(tmp_path, [{"id": "doc-1", "prompt": "one"}])
    output_path = tmp_path / "out.jsonl"
    plan_path = tmp_path / "plan.json"

    called = {"generate": 0}

    class FakeHallucinator:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            called["generate"] += 1
            raise AssertionError("plan-json must not call the LLM")

    monkeypatch.setattr("self_correct.cli.AntiHallucinator", FakeHallucinator)

    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path),
            "--output", str(output_path),
            "--plan-json", str(plan_path),
        ]
    )
    assert cmd_batch(args) == 0
    assert called["generate"] == 0
    assert not output_path.exists()
