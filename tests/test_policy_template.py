"""Tests for the starter verification-policy generator."""

from __future__ import annotations

import json
from dataclasses import fields

from self_correct.cli import _build_parser, _cmd_policy_template
from self_correct.core import VerificationPolicy


class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_starter_template_covers_the_full_schema() -> None:
    template = VerificationPolicy.starter_template()
    field_names = {field.name for field in fields(VerificationPolicy)}
    assert set(template) == field_names


def test_starter_template_round_trips_through_from_dict() -> None:
    policy = VerificationPolicy.from_dict(VerificationPolicy.starter_template())
    assert policy.min_verified_ratio == 0.9
    assert policy.max_flagged_claims == 2
    assert policy.max_critical_claims == 1
    assert policy.max_hallucination_density == 3.0
    assert policy.require_claims is True


def test_cmd_writes_a_valid_starter_policy(tmp_path, capsys) -> None:
    target = tmp_path / "policy.json"
    args = Args(output=str(target), force=False, stdout=False)
    assert _cmd_policy_template(args) == 0
    out = capsys.readouterr().out
    assert "Starter policy written" in out

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload) == {
        field.name for field in fields(VerificationPolicy)
    }
    VerificationPolicy.from_dict(payload)


def test_cmd_refuses_to_overwrite_without_force(tmp_path, capsys) -> None:
    target = tmp_path / "policy.json"
    target.write_text(json.dumps({"min_verified_ratio": 1.0}), encoding="utf-8")
    args = Args(output=str(target), force=False, stdout=False)
    assert _cmd_policy_template(args) == 1
    err = capsys.readouterr().err
    assert "already exists" in err
    # The existing file is untouched.
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "min_verified_ratio": 1.0
    }


def test_cmd_force_overwrites(tmp_path, capsys) -> None:
    target = tmp_path / "policy.json"
    target.write_text("{}", encoding="utf-8")
    args = Args(output=str(target), force=True, stdout=False)
    assert _cmd_policy_template(args) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "max_critical_claims" in payload


def test_cmd_stdout_prints_without_writing(tmp_path, capsys) -> None:
    args = Args(output=str(tmp_path / "never-written.json"), force=False, stdout=True)
    assert _cmd_policy_template(args) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload) == {field.name for field in fields(VerificationPolicy)}
    assert not (tmp_path / "never-written.json").exists()


def test_parser_accepts_positional_output_and_flags(tmp_path) -> None:
    args = _build_parser().parse_args(
        ["policy-template", str(tmp_path / "custom.json"), "--force"]
    )
    assert args.output == str(tmp_path / "custom.json")
    assert args.force is True
    assert args.stdout is False
