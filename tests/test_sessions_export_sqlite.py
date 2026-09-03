"""Tests for sessions.export_to_sqlite and the sessions-export-sqlite CLI."""

from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from self_correct import sessions
from self_correct.cli import main


def _save(tmp_path: Path, name: str, claims: list[dict]) -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt="verify me",
        config={"model": "gpt-4o-mini"},
        result={
            "status": "verified",
            "content": "",
            "verification_log": claims,
            "hallucinations_caught": [],
        },
    )
    return path


def test_export_to_sqlite_creates_expected_tables(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", claims=[
        {"claim": "Earth orbits the Sun", "is_valid": True, "critique": ""},
        {"claim": "Pluto is a planet", "is_valid": False, "critique": "Reclassified."},
    ])
    session = sessions.load_session(session_path)
    db_path = tmp_path / "out.db"
    rows = sessions.export_to_sqlite(db_path, session, table_name="claims")
    assert rows == 2
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT claim, is_valid, critique FROM claims ORDER BY id")
        rows = cur.fetchall()
    assert rows[0] == ("Earth orbits the Sun", 1, "")
    assert rows[1] == ("Pluto is a planet", 0, "Reclassified.")
    with sqlite3.connect(db_path) as conn:
        meta = conn.execute("SELECT session_prompt, status FROM claims_meta").fetchone()
    assert meta == ("verify me", "verified")


def test_export_to_sqlite_replaces_existing_file(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", claims=[
        {"claim": "x", "is_valid": True, "critique": ""},
    ])
    session = sessions.load_session(session_path)
    db_path = tmp_path / "out.db"
    sessions.export_to_sqlite(db_path, session)
    sessions.export_to_sqlite(db_path, session)
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM verification_log").fetchone()[0]
    assert count == 1


def test_export_to_sqlite_creates_parent_directory(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", claims=[])
    session = sessions.load_session(session_path)
    db_path = tmp_path / "nested" / "out.db"
    sessions.export_to_sqlite(db_path, session)
    assert db_path.exists()


def test_export_to_sqlite_rejects_unsafe_table_name(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", claims=[])
    session = sessions.load_session(session_path)
    with pytest.raises(ValueError, match="alphanumeric"):
        sessions.export_to_sqlite(tmp_path / "out.db", session, table_name="bad name")


def test_export_to_sqlite_handles_empty_log(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", claims=[])
    session = sessions.load_session(session_path)
    db_path = tmp_path / "out.db"
    rows = sessions.export_to_sqlite(db_path, session)
    assert rows == 0
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM verification_log").fetchone()[0]
    assert count == 0


def test_export_to_sqlite_skips_non_dict_entries(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", claims=[
        {"claim": "real", "is_valid": True, "critique": ""},
        "not a dict",
    ])
    session = sessions.load_session(session_path)
    db_path = tmp_path / "out.db"
    rows = sessions.export_to_sqlite(db_path, session)
    assert rows == 1


def test_cli_sessions_export_sqlite_writes_db(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", claims=[
        {"claim": "Sky is blue", "is_valid": True, "critique": ""},
    ])
    output = tmp_path / "out.db"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "sessions-export-sqlite",
            str(session_path),
            "--output",
            str(output),
        ])
    assert rc == 0
    assert output.exists()
    with sqlite3.connect(output) as conn:
        count = conn.execute("SELECT COUNT(*) FROM verification_log").fetchone()[0]
    assert count == 1


def test_cli_sessions_export_sqlite_returns_2_for_missing_session(tmp_path: Path) -> None:
    output = tmp_path / "out.db"
    rc = main([
        "sessions-export-sqlite",
        str(tmp_path / "missing.json"),
        "--output",
        str(output),
    ])
    assert rc == 2