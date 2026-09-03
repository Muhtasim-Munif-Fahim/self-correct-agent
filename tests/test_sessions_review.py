"""Tests for review.render_session_review and the sessions-review CLI."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from self_correct import review, sessions
from self_correct.cli import main


def _save(tmp_path: Path, name: str, prompt: str, claims: list[dict]) -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt=prompt,
        config={"model": "gpt-4o-mini"},
        result={
            "status": "verified" if all(c.get("is_valid") for c in claims) else "flagged",
            "content": "",
            "verification_log": claims,
            "hallucinations_caught": [c["claim"] for c in claims if not c.get("is_valid")],
        },
    )
    return path


def test_render_session_review_includes_sections() -> None:
    session = {
        "prompt": "Is the Earth round?",
        "config": {"model": "gpt-4o-mini"},
        "result": {
            "status": "flagged",
            "verification_log": [
                {"claim": "Earth is round", "is_valid": True, "critique": ""},
                {"claim": "Sky is green", "is_valid": False, "critique": "It is blue."},
            ],
        },
    }
    md = review.render_session_review(session)
    assert "# Session review" in md
    assert "## Provenance" in md
    assert "## Headline numbers" in md
    assert "## Top 1 flagged claims" in md
    assert "It is blue." in md
    assert "50.00%" in md
    assert "Distinct checks | 0" in md


def test_render_session_review_with_top_and_truncation() -> None:
    prompt = "x" * 500
    claims = [
        {"claim": f"c{i}", "is_valid": False, "critique": "x" * (10 - i) if i < 10 else "x"}
        for i in range(20)
    ]
    session = {
        "prompt": prompt,
        "config": {},
        "result": {"status": "flagged", "verification_log": claims},
    }
    md = review.render_session_review(session, top_n=3)
    assert "..." in md  # truncated prompt
    assert "## Top 3 flagged claims" in md
    assert md.count("Critique:") == 3


def test_render_session_review_includes_checks_seen() -> None:
    session = {
        "prompt": "p",
        "config": {},
        "result": {
            "status": "verified",
            "verification_log": [
                {"claim": "a", "is_valid": True, "critique": "", "check": "scientific"},
                {"claim": "b", "is_valid": True, "critique": "", "check": "factual"},
                {"claim": "c", "is_valid": True, "critique": "", "check": "scientific"},
            ],
        },
    }
    md = review.render_session_review(session)
    assert "## Distinct checks" in md
    assert "`scientific`" in md
    assert "`factual`" in md


def test_render_session_review_rejects_bad_session() -> None:
    with pytest.raises(ValueError, match="mapping"):
        review.render_session_review("not a mapping")  # type: ignore[arg-type]


def test_render_session_review_rejects_invalid_top_n() -> None:
    with pytest.raises(ValueError, match="top_n"):
        review.render_session_review({"result": {"verification_log": []}}, top_n=0)


def test_render_session_review_handles_empty_log() -> None:
    md = review.render_session_review({"prompt": "p", "config": {}, "result": {"status": "unknown", "verification_log": []}})
    assert "Total claims | 0" in md
    assert "No flagged claims" in md


def test_render_session_review_with_counts_returns_json_friendly_dict() -> None:
    session = {
        "prompt": "p",
        "config": {},
        "result": {
            "status": "verified",
            "verification_log": [
                {"claim": "a", "is_valid": True, "check": "scientific"},
                {"claim": "b", "is_valid": False, "check": "factual"},
            ],
        },
    }
    payload = review.render_session_review_with_counts(session)
    assert "markdown" in payload
    assert payload["counts"]["total_claims"] == 2
    assert payload["counts"]["verified"] == 1
    assert payload["counts"]["flagged"] == 1
    assert payload["counts"]["checks_seen"] == ["factual", "scientific"]


def test_cli_sessions_review_prints_markdown(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", "verify me", claims=[
        {"claim": "a", "is_valid": True, "critique": "", "check": "scientific"},
        {"claim": "b", "is_valid": False, "critique": "wrong", "check": "factual"},
    ])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["sessions-review", str(session_path)])
    assert rc == 0
    assert "# Session review" in buf.getvalue()


def test_cli_sessions_review_writes_output_file(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", "verify me", claims=[
        {"claim": "a", "is_valid": True, "critique": ""},
    ])
    output = tmp_path / "review.md"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["sessions-review", str(session_path), "--output", str(output)])
    assert rc == 0
    assert output.exists()
    assert "Review written to" in buf.getvalue()


def test_cli_sessions_review_json_mode(tmp_path: Path) -> None:
    session_path = _save(tmp_path, "session.json", "verify me", claims=[
        {"claim": "a", "is_valid": True, "critique": ""},
    ])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["sessions-review", str(session_path), "--json"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "markdown" in payload
    assert payload["counts"]["total_claims"] == 1


def test_cli_sessions_review_returns_2_for_missing_file(tmp_path: Path) -> None:
    rc = main(["sessions-review", str(tmp_path / "missing.json")])
    assert rc == 2