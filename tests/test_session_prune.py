"""Tests for pruning old saved session files."""

from __future__ import annotations

import json
import os
import time

from self_correct import cli, sessions
from self_correct.sessions import prune_sessions, save_session


def _write_session(tmp_path, name):
    path = tmp_path / name
    save_session(
        path, prompt="p", config={"model": "m"},
        result={"content": "x", "verification_log": []},
    )
    return path


def _age(path, seconds):
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_dry_run_lists_old_sessions_without_deleting(tmp_path) -> None:
    old = _write_session(tmp_path, "old.json")
    _age(old, 40 * 86400)

    result = prune_sessions([tmp_path], 30, dry_run=True)

    assert result["dry_run"] is True
    assert result["candidates"] == [str(old)]
    assert "deleted" not in result
    assert old.exists()


def test_delete_removes_only_old_valid_sessions(tmp_path) -> None:
    old = _write_session(tmp_path, "old.json")
    fresh = _write_session(tmp_path, "fresh.json")
    _age(old, 40 * 86400)

    result = prune_sessions([tmp_path], 30, dry_run=False)

    assert result["deleted"] == [str(old)]
    assert not old.exists()
    assert fresh.exists()
    assert result["kept"] == [str(fresh)]


def test_recent_sessions_are_never_candidates(tmp_path) -> None:
    path = _write_session(tmp_path, "recent.json")
    _age(path, 2 * 86400)

    result = prune_sessions([tmp_path], 30, dry_run=True)

    assert result["candidates"] == []
    assert path.exists()


def test_invalid_files_are_never_deleted(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    os.utime(broken, (time.time() - 100 * 86400, time.time() - 100 * 86400))
    old = _write_session(tmp_path, "old.json")
    _age(old, 40 * 86400)

    result = prune_sessions([tmp_path, broken], 30, dry_run=False)

    assert result["deleted"] == [str(old)]
    assert broken.exists()
    assert result["invalid"][0]["file"] == str(broken)


def test_fractional_days_are_supported(tmp_path) -> None:
    path = _write_session(tmp_path, "s.json")
    _age(path, 2 * 86400)

    assert prune_sessions([tmp_path], 1, dry_run=True)["candidates"] == [str(path)]
    assert prune_sessions([tmp_path], 10, dry_run=True)["candidates"] == []


def test_cli_dry_run_lists_candidates_without_deleting(tmp_path, capsys) -> None:
    old = _write_session(tmp_path, "old.json")
    _age(old, 40 * 86400)

    exit_code = cli.main(["sessions-prune", str(tmp_path), "--older-than", "30", "--dry-run"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Would prune 1 session(s)" in out
    assert str(old) in out
    assert old.exists()


def test_cli_delete_removes_files(tmp_path, capsys) -> None:
    old = _write_session(tmp_path, "old.json")
    _age(old, 40 * 86400)

    exit_code = cli.main(["sessions-prune", str(tmp_path), "--older-than", "30"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Pruned 1 session(s)" in out
    assert "Deleted 1 file(s)." in out
    assert not old.exists()


def test_cli_json_mode_reports_structure(tmp_path, capsys) -> None:
    old = _write_session(tmp_path, "old.json")
    _age(old, 40 * 86400)

    exit_code = cli.main(
        ["sessions-prune", str(tmp_path), "--older-than", "30", "--json", "--dry-run"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["candidates"] == [str(old)]


def test_cli_requires_a_positive_cutoff(tmp_path) -> None:
    _write_session(tmp_path, "s.json")
    try:
        cli.main(["sessions-prune", str(tmp_path), "--older-than", "0"])
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit for --older-than 0")
