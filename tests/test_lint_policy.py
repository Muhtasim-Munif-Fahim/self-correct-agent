import json
import pytest
from pathlib import Path
from self_correct.cli import cmd_config_lint_policy
from self_correct.core import VerificationPolicy

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def _write_policy(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path

def test_lint_policy_valid(tmp_path, capsys):
    path = _write_policy(tmp_path, 'policy.json', {'min_verified_ratio': 0.9})
    args = Args(policy=str(path), base=None, json=False)
    rc = cmd_config_lint_policy(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert 'valid' in out.lower()

def test_lint_policy_invalid_ratio(tmp_path, capsys):
    path = _write_policy(tmp_path, 'bad.json', {'min_verified_ratio': 1.5})
    args = Args(policy=str(path), base=None, json=False)
    rc = cmd_config_lint_policy(args)
    assert rc == 1
    out = capsys.readouterr().out
    assert 'ERROR' in out or 'error' in out.lower()

def test_lint_policy_missing_file(tmp_path, capsys):
    args = Args(policy=str(tmp_path / 'missing.json'), base=None, json=False)
    rc = cmd_config_lint_policy(args)
    assert rc == 2
    err = capsys.readouterr().err
    assert 'not found' in err.lower()

def test_lint_policy_layered(tmp_path, capsys):
    base = _write_policy(tmp_path, 'base.json', {'min_verified_ratio': 0.5})
    override = _write_policy(tmp_path, 'override.json', {'min_verified_ratio': 0.9})
    args = Args(policy=str(override), base=str(base), json=False)
    rc = cmd_config_lint_policy(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert 'Layer override' in out

def test_lint_policy_json_output(tmp_path, capsys):
    path = _write_policy(tmp_path, 'policy.json', {'min_verified_ratio': 0.9})
    args = Args(policy=str(path), base=None, json=True)
    rc = cmd_config_lint_policy(args)
    assert rc == 0
    out = capsys.readouterr().out
    import json as json_mod
    report = json_mod.loads(out)
    assert report['valid'] is True
    assert report['policy'] == str(path)

def test_lint_policy_warnings(tmp_path, capsys):
    # min_evidence_ratio > 0 but min_evidence_claims = 0
    path = _write_policy(tmp_path, 'policy.json', {'min_evidence_ratio': 0.5})
    args = Args(policy=str(path), base=None, json=False)
    rc = cmd_config_lint_policy(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert 'WARNING' in out
