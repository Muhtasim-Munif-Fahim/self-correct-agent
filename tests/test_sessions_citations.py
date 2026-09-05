"""Tests for collecting evidence sources cited across saved sessions."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from self_correct import citations, sessions
from self_correct.cli import _build_parser, main


def _save(tmp_path: Path, name: str, entries: list[dict]) -> Path:
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt="p",
        config={"model": "gpt-4o-mini"},
        result={
            "status": "verified",
            "content": "answer",
            "verification_log": entries,
            "hallucinations_caught": [],
        },
    )
    return path


def _entry(claim: str, sources: list[dict]) -> dict:
    return {
        "claim": claim,
        "is_valid": True,
        "critique": "",
        "evidence_sources": sources,
    }


def test_collect_deduplicates_sources_by_url() -> None:
    session = {
        "result": {
            "verification_log": [
                _entry("a", [{"title": "A", "url": "https://x", "tool": "Wiki"}]),
                _entry("b", [{"title": "A dup", "url": "https://x", "tool": "Other"}]),
                _entry("c", [{"title": "B", "url": "https://y", "tool": "Search"}]),
            ]
        }
    }
    sources = citations.collect_evidence_sources([session])
    assert [s["url"] for s in sources] == ["https://x", "https://y"]
    assert sources[0]["title"] == "A"


def test_collect_accepts_bare_result_payloads() -> None:
    result = {
        "verification_log": [
            _entry("a", [{"title": "A", "url": "https://x", "tool": "Wiki"}]),
        ]
    }
    sources = citations.collect_evidence_sources([result])
    assert sources == [{"title": "A", "url": "https://x", "tool": "Wiki"}]


def test_collect_skips_non_dict_entries_and_empty_urls() -> None:
    session = {
        "result": {
            "verification_log": [
                "not a dict",
                {"claim": "a", "is_valid": True, "evidence_sources": "nope"},
                _entry("b", [{"title": "", "url": "", "tool": "Wiki"}]),
                _entry("c", [{"url": "https://z", "tool": "Search"}]),
            ]
        }
    }
    sources = citations.collect_evidence_sources([session])
    assert len(sources) == 1
    assert sources[0]["url"] == "https://z"
    assert sources[0]["title"] == ""


def test_to_text_lists_numbered_sources() -> None:
    out = citations.to_text([
        {"title": "Wiki A", "url": "https://a", "tool": "Wikipedia"},
    ])
    assert out == "1. Wiki A \u2014 https://a (via Wikipedia)\n"


def test_to_text_returns_empty_for_no_sources() -> None:
    assert citations.to_text([]) == ""


def test_to_json_produces_a_list() -> None:
    out = citations.to_json([{"title": "T", "url": "https://t", "tool": "Wiki"}])
    parsed = json.loads(out)
    assert parsed == [{"title": "T", "url": "https://t", "tool": "Wiki"}]


def test_to_bibtex_emits_misc_entries_with_url_and_unique_citekeys() -> None:
    sources = [
        {"title": "Alpha", "url": "https://a", "tool": "Wikipedia"},
        {"title": "Alpha", "url": "https://b", "tool": "Search"},
        {"url": "https://c", "tool": "Archive"},
    ]
    out = citations.to_bibtex(sources)
    assert out.count("@misc{") == 3
    assert "url = {https://a}" in out
    assert "url = {https://b}" in out
    assert "url = {https://c}" in out
    citekeys = [line.split("{", 1)[1].split(",", 1)[0] for line in out.splitlines() if line.startswith("@misc{")]
    assert len(set(citekeys)) == 3


def test_format_sources_dispatches_on_fmt() -> None:
    sources = [{"title": "T", "url": "https://t", "tool": "Wiki"}]
    assert citations.format_sources(sources, "json").startswith("[")
    assert citations.format_sources(sources, "bibtex").startswith("@misc{")
    assert citations.format_sources(sources, "text").startswith("1.")
    assert citations.format_sources(sources, "text") == citations.to_text(sources)


def test_sessions_collect_citations_scans_and_dedupes(tmp_path: Path) -> None:
    _save(tmp_path, "a.json", [
        _entry("a", [{"title": "A", "url": "https://x", "tool": "Wiki"}]),
    ])
    _save(tmp_path, "b.json", [
        _entry("b", [{"title": "A too", "url": "https://x", "tool": "Wiki"}]),
        _entry("c", [{"title": "B", "url": "https://y", "tool": "Search"}]),
    ])
    result = sessions.collect_citations([tmp_path])
    assert result["scanned"] == 2
    assert result["invalid"] == []
    assert [s["url"] for s in result["sources"]] == ["https://x", "https://y"]


def test_sessions_collect_citations_reports_invalid_files(tmp_path: Path) -> None:
    good = _save(tmp_path, "good.json", [
        _entry("a", [{"title": "A", "url": "https://x", "tool": "Wiki"}]),
    ])
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    result = sessions.collect_citations([broken, good])
    assert result["scanned"] == 1
    assert result["invalid"][0]["file"] == str(broken)
    assert result["sources"] == [{"title": "A", "url": "https://x", "tool": "Wiki"}]


def test_cli_citations_default_text(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [
        _entry("a", [{"title": "A", "url": "https://x", "tool": "Wiki"}]),
    ])
    rc = main(["sessions-citations", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1. A" in out
    assert "https://x" in out
    assert "via Wiki" in out


def test_cli_citations_json_and_bibtex(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [
        _entry("a", [{"title": "A", "url": "https://x", "tool": "Wiki"}]),
    ])
    assert main(["sessions-citations", "--format", "bibtex", str(tmp_path)]) == 0
    assert "@misc{" in capsys.readouterr().out

    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["sessions-citations", "--format", "json", str(tmp_path)])
    parsed = json.loads(buf.getvalue())
    assert parsed[0]["url"] == "https://x"


def test_cli_citations_writes_output_file(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "a.json", [
        _entry("a", [{"title": "A", "url": "https://x", "tool": "Wiki"}]),
    ])
    out = tmp_path / "nested" / "refs.bib"
    rc = main(["sessions-citations", "--format", "bibtex", "--output", str(out), str(tmp_path)])
    assert rc == 0
    assert out.exists()
    assert "@misc{" in out.read_text(encoding="utf-8")
    assert "Wrote 1 source(s)" in capsys.readouterr().out


def test_cli_citations_returns_2_when_no_valid_sessions(tmp_path: Path, capsys) -> None:
    rc = main(["sessions-citations", str(tmp_path / "missing.json")])
    assert rc == 2
    assert "No valid session files found." in capsys.readouterr().err


def test_cli_citations_empty_when_no_sources(tmp_path: Path, capsys) -> None:
    _save(tmp_path, "empty.json", [])
    rc = main(["sessions-citations", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out == ""
