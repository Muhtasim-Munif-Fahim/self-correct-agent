"""Tests for the history label feature."""

from __future__ import annotations

import json

import pytest

from self_correct import history


def _runs(*entries: dict) -> list[dict]:
    out = []
    for entry in entries:
        out.append({"timestamp": 1.0, **entry})
    return out


def test_parse_label_string_comma_split() -> None:
    assert history.parse_label("nightly,smoke") == ["nightly", "smoke"]


def test_parse_label_strips_whitespace() -> None:
    assert history.parse_label(" alpha , beta ") == ["alpha", "beta"]


def test_parse_label_accepts_list() -> None:
    assert history.parse_label(["alpha", "beta"]) == ["alpha", "beta"]


def test_parse_label_handles_none() -> None:
    assert history.parse_label(None) == []


def test_parse_label_rejects_empty_entries() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        history.parse_label("alpha,,beta")


def test_parse_label_rejects_non_strings() -> None:
    with pytest.raises(ValueError, match="strings"):
        history.parse_label(["alpha", 2])  # type: ignore[list-item]


def test_parse_label_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="string or a list"):
        history.parse_label(42)  # type: ignore[arg-type]


def test_filter_runs_label_requires_all_tags() -> None:
    runs = _runs(
        {"label": ["alpha", "beta"]},
        {"label": ["alpha"]},
        {"label": ["beta"]},
        {},
    )
    matched = history.filter_runs(runs, label="alpha,beta")
    assert len(matched) == 1
    assert matched[0]["label"] == ["alpha", "beta"]


def test_filter_runs_label_accepts_list_form() -> None:
    runs = _runs({"label": ["a", "b"]}, {"label": ["a"]})
    matched = history.filter_runs(runs, label=["a", "b"])
    assert len(matched) == 1


def test_filter_runs_model_filter() -> None:
    runs = _runs({"model": "gpt-4o-mini"}, {"model": "gpt-4o"})
    matched = history.filter_runs(runs, model="gpt-4o")
    assert len(matched) == 1


def test_filter_runs_combined_label_and_model() -> None:
    runs = _runs(
        {"model": "gpt-4o", "label": ["a", "b"]},
        {"model": "gpt-4o", "label": ["a"]},
        {"model": "gpt-4o-mini", "label": ["a", "b"]},
    )
    matched = history.filter_runs(runs, label="a,b", model="gpt-4o")
    assert len(matched) == 1


def test_aggregate_includes_label_counts() -> None:
    runs = _runs(
        {"label": ["alpha", "shared"]},
        {"label": ["beta", "shared"]},
        {"label": ["shared"]},
        {"label": "alpha"},
        {"model": "gpt-4o", "duration": 0.1, "claims": 1, "claims_verified": 1},
    )
    summary = history.aggregate(runs)
    assert summary["labels"] == {"shared": 3, "alpha": 2, "beta": 1}


def test_record_run_persists_label(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    history.record_run({"command": "verify", "label": ["alpha", "beta"]}, path=path)
    history.record_run({"command": "verify", "label": "gamma"}, path=path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["label"] == ["alpha", "beta"]
    assert records[1]["label"] == ["gamma"]


def test_filter_runs_skips_malformed_label_entries() -> None:
    runs = _runs({"label": "  "}, {"label": ["alpha"]})
    matched = history.filter_runs(runs, label="alpha")
    assert len(matched) == 1
    # No crash on a malformed label that parse_label would normally reject.
    matched = history.filter_runs(runs, label="ghost")
    assert matched == []