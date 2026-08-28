"""CSV export of per-claim verification verdicts."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, Iterable, List, Mapping

from .core import classify_severity

#: Column order and names of the exported CSV.
CSV_COLUMNS = ["id", "claim", "verdict", "severity", "critique", "evidence_used", "cached"]


def _result_from(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the embedded result of a session payload, or the payload itself."""

    result = payload.get("result")
    return result if isinstance(result, dict) else payload


def verdict_rows(payload: Mapping[str, Any]) -> Iterable[Dict[str, str]]:
    """Yield one CSV row per claim verdict in a session or result.

    Each row carries the claim text, its verdict, the severity class of the
    critique for flagged claims, whether the check used external evidence,
    and whether the verdict came from the cache. Phase metadata entries and
    budget-halt markers are skipped, mirroring the JUnit export.
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


def result_to_csv(payload: Mapping[str, Any]) -> str:
    """Render per-claim verdicts as CSV text with a header row."""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in verdict_rows(payload):
        writer.writerow(row)
    return buffer.getvalue()
