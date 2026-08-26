"""Tests for layered verification policies (base + override files)."""

from __future__ import annotations

import json
import re

import pytest

from self_correct.cli import _resolve_policy
from self_correct.core import load_layered_policy


def _write_policy(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_override_replaces_base_values(tmp_path) -> None:
    base = _write_policy(tmp_path, "base.json", {"min_verified_ratio": 0.5})
    override = _write_policy(tmp_path, "strict.json", {"min_verified_ratio": 0.9})

    policy, conflicts = load_layered_policy([base, override])

    assert policy.min_verified_ratio == 0.9
    assert len(conflicts) == 1
    assert "min_verified_ratio" in conflicts[0]
    assert "strict.json" in conflicts[0]


def test_fields_absent_from_override_keep_base_value(tmp_path) -> None:
    base = _write_policy(tmp_path, "base.json", {"max_flagged_claims": 2, "require_claims": True})
    override = _write_policy(tmp_path, "override.json", {"min_verified_claims": 3})

    policy, conflicts = load_layered_policy([base, override])

    assert policy.max_flagged_claims == 2
    assert policy.require_claims is True
    assert policy.min_verified_claims == 3
    assert conflicts == []


def test_identical_values_are_not_reported_as_conflicts(tmp_path) -> None:
    base = _write_policy(tmp_path, "base.json", {"max_flagged_claims": 1})
    override = _write_policy(tmp_path, "override.json", {"max_flagged_claims": 1})

    _, conflicts = load_layered_policy([base, override])

    assert conflicts == []


def test_merged_result_is_validated_once(tmp_path) -> None:
    base = _write_policy(tmp_path, "base.json", {})
    override = _write_policy(tmp_path, "bad.json", {"no_such_field": 1})

    with pytest.raises(ValueError, match="unknown verification policy fields"):
        load_layered_policy([base, override])


def test_invalid_ratio_in_override_is_rejected(tmp_path) -> None:
    base = _write_policy(tmp_path, "base.json", {"min_verified_ratio": 0.5})
    override = _write_policy(tmp_path, "bad.json", {"min_verified_ratio": 1.5})

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_layered_policy([base, override])


def test_unreadable_layer_names_the_offending_file(tmp_path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(ValueError, match=re.escape(str(missing))):
        load_layered_policy([missing])


def test_resolver_passes_a_single_file_through(tmp_path) -> None:
    path = _write_policy(tmp_path, "only.json", {"max_critical_claims": 4})

    policy = _resolve_policy(str(path))

    assert policy is not None
    assert policy.max_critical_claims == 4


def test_resolver_reports_conflicts_for_layered_files(tmp_path, capsys) -> None:
    base = _write_policy(tmp_path, "base.json", {"max_flagged_claims": 2})
    override = _write_policy(tmp_path, "strict.json", {"max_flagged_claims": 0})

    policy = _resolve_policy([str(base), str(override)])

    err = capsys.readouterr().err
    assert policy is not None
    assert policy.max_flagged_claims == 0
    assert "policy override: max_flagged_claims: 2 overridden with 0" in err


def test_resolver_ignores_empty_values() -> None:
    assert _resolve_policy(None) is None
    assert _resolve_policy("") is None
    assert _resolve_policy([]) is None