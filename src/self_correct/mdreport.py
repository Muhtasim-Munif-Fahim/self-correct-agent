"""Markdown table export of per-claim verification verdicts.

The per-claim verdicts are already available as CSV (see :mod:`csvreport`),
JSONL (see :mod:`jsonlreport`) and JUnit XML (see :mod:`junit`); this module
adds a GitHub-flavored Markdown table, the format that drops most cleanly into
Notion, wikis and review prose. It mirrors those modules exactly: phase
metadata entries and verdicts without a claim text are skipped, and the verdict
column is a plain "verified"/"flagged" string.

Cell contents are escaped so a claim or critique containing a pipe character
or a newline cannot break the table grid.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from .core import classify_severity

#: Column order printed in the Markdown table, kept identical to
#: ``csvreport.CSV_COLUMNS`` so every per-claim export lines up.
MD_COLUMNS = ["id", "claim", "verdict", "severity", "critique", "evidence_used", "cached"]


def _result_from(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the embedded result of a session payload, or the payload itself."""

    result = payload.get("result")
    return result if isinstance(result, dict) else payload


def _md_escape(value: str) -> str:
    """Escape pipe characters and newlines so a cell stays on one grid line."""

    return value.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def verdict_rows(payload: Mapping[str, Any]) -> Iterable[Dict[str, str]]:
    """Yield one dict per claim verdict in a session or result.

    Mirrors :func:`csvreport.verdict_rows` and :func:`jsonlreport.verdict_records`
    so a Markdown export stays in lock-step with the CSV and JSONL exports:
    each verdict carries the claim text, verdict, severity class of a flagged
    critique, whether the check used external evidence, and whether the
    verdict came from the cache. Booleans render as ``true``/``false`` text,
    matching the CSV export.
    """

    result = _result_from(payload)
    for idx, entry in enumerate(result.get("verification_log") or [], 1):
        if not isinstance(entry, dict):
            continue
        if "is_valid" not in entry or not entry.get("claim"):
            continue
        critique = str(entry.get("critique", ""))
        flagged = not bool(entry["is_valid"])
        yield {
            "id": str(idx),
            "claim": str(entry["claim"]),
            "verdict": "flagged" if flagged else "verified",
            "severity": classify_severity(critique) if flagged else "",
            "critique": critique,
            "evidence_used": str(bool(entry.get("evidence_used"))).lower(),
            "cached": str(bool(entry.get("cached"))).lower(),
        }


def result_to_markdown(payload: Mapping[str, Any]) -> str:
    """Render per-claim verdicts as a GitHub-flavored Markdown table.

    The header and separator are always present, even when there are no verdict
    rows, so the output is a well-formed (if empty) Markdown table.
    """

    rows = list(verdict_rows(payload))
    lines: List[str] = []
    lines.append("| " + " | ".join(MD_COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in MD_COLUMNS) + " |")
    for row in rows:
        cells = [_md_escape(str(row.get(col, ""))) for col in MD_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


__all__ = [
    "MD_COLUMNS",
    "verdict_rows",
    "result_to_markdown",
]
