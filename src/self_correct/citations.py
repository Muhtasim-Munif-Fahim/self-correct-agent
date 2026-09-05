"""Collect evidence sources cited during verification.

Each claim verdict can carry ``evidence_sources`` — the titles, URLs and tool
names retrieved while a claim was checked. The per-claim export modules
(CSV, JSONL, JUnit, SQLite) deliberately drop that field, so the only way to
audit which sources actually backed a verification was to read the raw log.

This module rescues that evidence trail: it scans session/result payloads,
deduplicates sources by URL, and renders them as JSON, plain text or BibTeX so
a reviewer can follow every link without re-running the model.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List

#: Cite-key suffix length taken from a SHA-256 of the URL.
_URL_HASH_LEN = 8

#: Flag chars that are illegal unescaped inside a BibTeX field body.
_BIBTEX_SPECIAL = re.compile(r"[{}]")


def _bibtex_escape(text: str) -> str:
    """Escape backslashes and braces for a BibTeX field body."""

    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _bibtex_citekey(source: Dict[str, str], index: int) -> str:
    """Build a stable, unique cite key from a title slug and URL hash."""

    slug = re.sub(r"[^a-z0-9]+", "", (source.get("title") or "").lower())[:20]
    if not slug:
        slug = f"source{index}"
    digest = hashlib.sha256(source["url"].encode("utf-8")).hexdigest()[:_URL_HASH_LEN]
    return f"{slug}{digest}"


class EvidenceSource:
    """A single deduplicated evidence source referenced by a verified claim."""

    __slots__ = ("title", "url", "tool")

    def __init__(self, title: str, url: str, tool: str) -> None:
        self.title = title
        self.url = url
        self.tool = tool

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.url, "tool": self.tool}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EvidenceSource) and self.url == other.url

    def __hash__(self) -> int:
        return hash(self.url)

    def __repr__(self) -> str:
        return f"EvidenceSource(url={self.url!r})"


def collect_evidence_sources(
    payloads: Iterable[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Return deduplicated evidence sources from session/result payloads.

    Each payload may be a full session (carrying a ``result`` key) or a bare
    result object. Sources are pulled from every verdict entry's
    ``evidence_sources`` list and deduplicated by URL, keeping the first
    sighting so output order is stable. Non-dict entries, missing URLs and
    duplicate URLs are silently skipped; an empty URL is never recorded.
    """

    sources: List[Dict[str, str]] = []
    seen_urls: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        if not isinstance(result, dict):
            result = payload
        for entry in result.get("verification_log") or []:
            if not isinstance(entry, dict):
                continue
            for source in entry.get("evidence_sources") or []:
                if not isinstance(source, dict):
                    continue
                url = str(source.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                sources.append({
                    "title": str(source.get("title", "")).strip(),
                    "url": url,
                    "tool": str(source.get("tool", "")).strip(),
                })
    return sources


def to_text(sources: List[Dict[str, str]]) -> str:
    """Render sources as a numbered list: title — url (via tool)."""

    if not sources:
        return ""
    lines: List[str] = []
    for index, source in enumerate(sources, 1):
        title = source["title"] or source["url"]
        tool = source["tool"] or "tool"
        lines.append(f"{index}. {title} \u2014 {source['url']} (via {tool})")
    return "\n".join(lines) + "\n"


def to_json(sources: List[Dict[str, str]]) -> str:
    """Render sources as a JSON array."""

    return json.dumps(sources, indent=2, ensure_ascii=False) + "\n"


def to_bibtex(sources: List[Dict[str, str]]) -> str:
    """Render sources as BibTeX ``@misc`` entries, one per source."""

    if not sources:
        return ""
    entries: List[str] = []
    for index, source in enumerate(sources, 1):
        citekey = _bibtex_citekey(source, index)
        title = source.get("title") or "untitled"
        url = source.get("url", "")
        tool = source.get("tool") or "unknown"
        entries.append(
            f"@misc{{{citekey},\n"
            f"  title = {{{_bibtex_escape(title)}}},\n"
            f"  url = {{{_bibtex_escape(url)}}},\n"
            f"  howpublished = {{Retrieved via {tool}}}\n"
            "}"
        )
    return "\n\n".join(entries) + "\n"


def format_sources(sources: List[Dict[str, str]], fmt: str = "text") -> str:
    """Render collected sources in one of ``text``, ``json`` or ``bibtex``."""

    if fmt == "json":
        return to_json(sources)
    if fmt == "bibtex":
        return to_bibtex(sources)
    return to_text(sources)


__all__ = [
    "EvidenceSource",
    "collect_evidence_sources",
    "to_text",
    "to_json",
    "to_bibtex",
    "format_sources",
]
