"""Tests for severity-weighted scoring in the batch index."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from self_correct.cli import _build_batch_index, _build_parser, _severity_counts, cmd_batch


def _record(id, *, flagged=0, error=None, severities=None):
    record = {
        "id": id,
        "prompt": "p",
        "hallucinations_caught": ["bad"] * flagged,
        "verification_log": [
            {"claim": "c", "is_valid": False, "critique": "Claim is incorrect"}
            if i < flagged
            else {"claim": "c", "is_valid": True}
            for i in range(flagged + 1)
        ],
    }
    if severities is not None:
        record["severity_summary"] = severities
    if error:
        record["error"] = error
    return record


def test_counts_prefer_the_pipeline_severity_summary() -> None:
    record = _record("a", flagged=2, severities={"critical": 1, "major": 1, "minor": 0})
    assert _severity_counts(record) == {"critical": 1, "major": 1, "minor": 0}


def test_counts_classify_log_critiques_when_summary_is_missing() -> None:
    record = {
        "id": "x",
        "hallucinations_caught": ["unused"],
        "verification_log": [
            {"claim": "a", "is_valid": False, "critique": "The figure is fabricated"},
            {"claim": "b", "is_valid": False, "critique": "Cannot be verified"},
            {"claim": "c", "is_valid": True},
        ],
    }
    assert _severity_counts(record) == {"critical": 1, "major": 1, "minor": 0}


def test_counts_fall_back_to_hallucinations_text() -> None:
    record = {
        "id": "x",
        "hallucinations_caught": ["This contradicts the source", "vague wording"],
    }
    assert _severity_counts(record) == {"critical": 1, "major": 0, "minor": 1}


def test_index_aggregates_weighted_scores() -> None:
    results = [
        _record("a", flagged=1, severities={"critical": 1, "major": 0, "minor": 0}),
        _record("b", flagged=2, severities={"critical": 0, "major": 1, "minor": 1}),
        _record("c"),
    ]
    index = _build_batch_index(results)
    by_id = {item["id"]: item for item in index["items"]}

    assert by_id["a"]["severity_counts"] == {"critical": 1, "major": 0, "minor": 0}
    assert by_id["a"]["severity_score"] == 3
    assert by_id["b"]["severity_score"] == 3  # major 2 + minor 1
    assert by_id["c"]["severity_score"] == 0

    assert index["severity"] == {
        "totals": {"critical": 1, "major": 1, "minor": 1},
        "score": 6,
        "weights": {"critical": 3, "major": 2, "minor": 1},
    }


def test_error_items_score_zero() -> None:
    index = _build_batch_index([_record("x", error="RuntimeError: down")])
    item = index["items"][0]
    assert item["status"] == "error"
    assert item["severity_counts"] == {"critical": 0, "major": 0, "minor": 0}
    assert item["severity_score"] == 0
    assert index["severity"]["score"] == 0


def test_empty_results_score_zero() -> None:
    index = _build_batch_index([])
    assert index["severity"] == {
        "totals": {"critical": 0, "major": 0, "minor": 0},
        "score": 0,
        "weights": {"critical": 3, "major": 2, "minor": 1},
    }


def _install_fake_pipeline(monkeypatch, responses):
    """Replace the batch pipeline with canned to_dict() results."""

    fake_module = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    class FakeHallucinator:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return SimpleNamespace(to_dict=lambda: responses.pop(0))

    monkeypatch.setattr("self_correct.cli.AntiHallucinator", FakeHallucinator)


def test_cmd_batch_reports_the_batch_severity_score(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        json.dumps({"id": "doc-1", "prompt": "one"}) + "\n",
        encoding="utf-8",
    )
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
                "severity_summary": {"critical": 1, "major": 0, "minor": 0},
            }
        ],
    )
    args = _build_parser().parse_args(
        ["batch", "--input", str(input_path), "--format", "jsonl"]
    )
    cmd_batch(args)
    err = capsys.readouterr().err
    assert "Severity score: 3 (critical 1, major 0, minor 0)" in err


def test_cmd_batch_index_carries_severity_block(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        json.dumps({"id": "doc-1", "prompt": "one"}) + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"
    _install_fake_pipeline(
        monkeypatch,
        [
            {
                "id": "doc-1",
                "content": "text",
                "hallucinations_caught": ["bad"],
                "severity_summary": {"critical": 0, "major": 0, "minor": 1},
            }
        ],
    )
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path),
            "--index", str(index_path), "--format", "jsonl",
        ]
    )
    cmd_batch(args)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["severity"]["totals"] == {"critical": 0, "major": 0, "minor": 1}
    assert index["severity"]["score"] == 1
    assert index["items"][0]["severity_score"] == 1
