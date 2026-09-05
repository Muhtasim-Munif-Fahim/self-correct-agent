"""Tests for bundling saved sessions into a portable archive."""

from __future__ import annotations

import io
import json
import tarfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from self_correct import bundle, sessions
from self_correct.cli import _build_parser, main


def _save(tmp_path: Path, name: str, log: list[dict]) -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt="p",
        config={"model": "gpt-4o-mini"},
        result={
            "status": "flagged" if any(not e.get("is_valid") for e in log if isinstance(e, dict)) else "verified",
            "content": "answer",
            "verification_log": log,
            "hallucinations_caught": [],
        },
    )
    return path


def _archive_names(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path, "r:gz") as archive:
        return archive.getnames()


def _read_member_text(tar_path: Path, member: str) -> str:
    with tarfile.open(tar_path, "r:gz") as archive:
        extracted = archive.extractfile(member)
        assert extracted is not None
        return extracted.read().decode("utf-8")


def test_create_bundle_writes_manifest_and_sessions(tmp_path: Path) -> None:
    path = _save(tmp_path, "a.json", [
        {"claim": "Earth is round", "is_valid": True, "critique": "", "evidence_sources": []},
        {"claim": "sky is green", "is_valid": False, "critique": "Statement is false.", "evidence_sources": [
            {"title": "Wiki", "url": "https://x", "tool": "Wikipedia"}
        ]},
    ])
    archive = tmp_path / "out.tar.gz"
    result = bundle.create_bundle([path], archive)

    assert result["sessions"] == 1
    assert result["invalid"] == []
    assert result["compressed"] is True
    assert archive.exists()
    names = _archive_names(archive)
    assert "manifest.json" in names
    assert "sessions/0001_a.json" in names

    manifest = json.loads(_read_member_text(archive, "manifest.json"))
    assert manifest["schema_version"] == 1
    assert manifest["total_sessions"] == 1
    assert manifest["total_claims"] == 2
    assert manifest["total_flagged"] == 1
    entry = manifest["sessions"][0]
    assert entry["name"] == "a.json"
    assert entry["model"] == "gpt-4o-mini"
    assert entry["status"] == "flagged"
    assert entry["claims"] == 2
    assert entry["verified"] == 1
    assert entry["flagged"] == 1
    assert entry["severities"]["critical"] == 1
    assert entry["evidence_sources"] == 1


def test_create_bundle_member_can_be_reloaded(tmp_path: Path) -> None:
    path = _save(tmp_path, "s.json", [
        {"claim": "Earth is round", "is_valid": True, "critique": ""},
    ])
    archive = tmp_path / "out.tar.gz"
    bundle.create_bundle([path], archive)

    payload = json.loads(_read_member_text(archive, "sessions/0001_s.json"))
    assert sessions.load_session(Path(str(path))) is not None
    assert payload["result"]["verification_log"][0]["claim"] == "Earth is round"


def test_create_bundle_no_compress_writes_plain_tar(tmp_path: Path) -> None:
    path = _save(tmp_path, "a.json", [{"claim": "x", "is_valid": True, "critique": ""}])
    archive = tmp_path / "out.tar"
    result = bundle.create_bundle([path], archive, compress=False)

    assert result["compressed"] is False
    with tarfile.open(archive, "r:") as archive_obj:
        assert "manifest.json" in archive_obj.getnames()


def test_create_bundle_expands_directories(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", [{"claim": "x", "is_valid": True, "critique": ""}])
    _save(tmp_path, "b.json", [{"claim": "y", "is_valid": True, "critique": ""}])
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    archive = tmp_path / "out.tar.gz"
    bundle.create_bundle([tmp_path], archive)
    manifest = json.loads(_read_member_text(archive, "manifest.json"))
    assert manifest["total_sessions"] == 2


def test_create_bundle_dedupes_overlapping_paths(tmp_path: Path) -> None:
    path = _save(tmp_path, "a.json", [{"claim": "x", "is_valid": True, "critique": ""}])
    archive = tmp_path / "out.tar.gz"
    bundle.create_bundle([path, path], archive)
    manifest = json.loads(_read_member_text(archive, "manifest.json"))
    assert manifest["total_sessions"] == 1


def test_create_bundle_reports_invalid_files_without_failing(tmp_path: Path) -> None:
    good = _save(tmp_path, "good.json", [{"claim": "x", "is_valid": True, "critique": ""}])
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    archive = tmp_path / "out.tar.gz"
    result = bundle.create_bundle([tmp_path], archive)

    assert result["sessions"] == 1
    assert len(result["invalid"]) == 1
    assert result["invalid"][0]["file"] == str(tmp_path / "broken.json")
    manifest = json.loads(_read_member_text(archive, "manifest.json"))
    assert len(manifest["invalid"]) == 1


def test_create_bundle_raises_when_no_session_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no session files"):
        bundle.create_bundle([tmp_path], tmp_path / "out.tar.gz")


def test_create_bundle_member_names_are_unique(tmp_path: Path) -> None:
    sub_a = tmp_path / "a"
    sub_b = tmp_path / "b"
    sub_a.mkdir()
    sub_b.mkdir()
    _save(sub_a, "session.json", [{"claim": "x", "is_valid": True, "critique": ""}])
    _save(sub_b, "session.json", [{"claim": "y", "is_valid": True, "critique": ""}])

    archive = tmp_path / "out.tar.gz"
    bundle.create_bundle([sub_a, sub_b], archive)
    names = [n for n in _archive_names(archive) if n.startswith("sessions/")]
    assert len(names) == 2
    assert len(set(names)) == 2


def test_cli_bundle_writes_archive(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [{"claim": "x", "is_valid": True, "critique": ""}])
    archive = tmp_path / "out.tar.gz"

    rc = main(["sessions-bundle", str(tmp_path), "--output", str(archive)])
    out = capsys.readouterr().out

    assert rc == 0
    assert archive.exists()
    assert "Bundled 1 session(s)" in out
    assert "manifest.json" in _archive_names(archive)


def test_cli_bundle_json_output(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [{"claim": "x", "is_valid": True, "critique": ""}])
    archive = tmp_path / "out.tar.gz"

    rc = main(["sessions-bundle", "--json", str(tmp_path), "--output", str(archive)])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["sessions"] == 1


def test_cli_bundle_returns_2_when_no_valid_sessions(tmp_path: Path, capsys) -> None:
    (tmp_path / "broken.json").write_text("[]", encoding="utf-8")
    archive = tmp_path / "out.tar.gz"

    rc = main(["sessions-bundle", str(tmp_path), "--output", str(archive)])
    err = capsys.readouterr().err

    assert rc == 2
    assert "No valid session files found." in err


def test_cli_bundle_requires_output(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [{"claim": "x", "is_valid": True, "critique": ""}])
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["sessions-bundle", str(tmp_path)])
