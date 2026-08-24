"""Tests for the batch summary index."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from self_correct.cli import _build_batch_index, _build_parser, cmd_batch


def _record(id, *, flagged=False, error=None, verified=1, total=2):
    record = {
        "id": id,
        "prompt": "p",
        "hallucinations_caught": ["bad"] if flagged else [],
        "claim_summary": {
            "total_claims": total,
            "verified_claims": verified,
            "flagged_claims": total - verified,
            "evidence_claims": 0,
        },
    }
    if error:
        record["error"] = error
        del record["claim_summary"]
    return record


def test_index_classifies_clean_flagged_and_error_items() -> None:
    results = [
        _record("a"),
        _record("b", flagged=True, verified=1, total=2),
        _record("c", error="RuntimeError: down"),
    ]
    index = _build_batch_index(results)

    assert index["summary"] == {"total": 3, "clean": 1, "flagged": 1, "error": 1}
    statuses = [item["status"] for item in index["items"]]
    assert statuses == ["clean", "flagged", "error"]
    assert [item["position"] for item in index["items"]] == [0, 1, 2]


def test_error_items_carry_the_message_without_claim_counts() -> None:
    index = _build_batch_index([_record("c", error="ValueError: no key")])
    item = index["items"][0]
    assert item["error"] == "ValueError: no key"
    assert "claims_total" not in item


def test_counts_fall_back_to_the_verification_log() -> None:
    record = {
        "id": "x",
        "verification_log": [
            {"claim": "a", "is_valid": True},
            {"phase": "extraction", "warning": "none"},
            {"claim": "b", "is_valid": False},
        ],
        "hallucinations_caught": ["bad"],
    }
    item = _build_batch_index([record])["items"][0]
    assert item["claims_total"] == 2
    assert item["claims_verified"] == 1
    assert item["flagged_count"] == 1


def test_empty_results_produce_a_zeroed_index() -> None:
    assert _build_batch_index([])["summary"]["total"] == 0


def test_parser_accepts_index_path() -> None:
    args = _build_parser().parse_args(
        ["batch", "--input", "in.jsonl", "--index", "idx.json"]
    )
    assert args.index == "idx.json"


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


def test_cmd_batch_writes_results_and_index(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        json.dumps({"id": "doc-1", "prompt": "one"}) + "\n"
        + json.dumps({"id": "doc-2", "prompt": "two"}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    index_path = tmp_path / "index.json"

    _install_fake_pipeline(
        monkeypatch,
        [
            dict(_record("doc-1"), content="clean text"),
            _record("doc-2", flagged=True),
        ],
    )
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path), "--output", str(output_path),
            "--index", str(index_path), "--format", "jsonl",
        ]
    )
    cmd_batch(args)
    err = capsys.readouterr().err
    assert "Batch index written" in err

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["summary"] == {"total": 2, "clean": 1, "flagged": 1, "error": 0}
    by_id = {item["id"]: item for item in index["items"]}
    assert by_id["doc-1"]["position"] == 0
    assert by_id["doc-2"]["position"] == 1
    assert by_id["doc-2"]["status"] == "flagged"


def test_cmd_batch_json_format_positions_are_array_indices(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps({"id": "only", "prompt": "p"}) + "\n", encoding="utf-8")
    output_path = tmp_path / "out.json"

    _install_fake_pipeline(monkeypatch, [dict(_record("only"), content="text")])
    args = _build_parser().parse_args(
        [
            "batch", "--input", str(input_path), "--output", str(output_path),
            "--index", str(tmp_path / "index.json"), "--format", "json",
        ]
    )
    cmd_batch(args)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 1

    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["items"][0]["position"] == 0
