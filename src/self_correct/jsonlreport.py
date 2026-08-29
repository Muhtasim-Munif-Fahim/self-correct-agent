"""JSONL export of per-claim verification verdicts."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping

from .core import classify_severity


def _result_from(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the embedded result of a session payload, or the payload itself."""

    result = payload.get("result")
    return result if isinstance(result, dict) else payload


def verdict_records(payload: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield one JSON object per claim verdict in a session or result.

    Each record carries the claim text, its verdict, the severity class of
    the critique for flagged claims, whether the check used external
    evidence, and whether the verdict came from the cache. Booleans stay
    native booleans so downstream JSON tooling can filter on them directly.
    Phase metadata entries and budget-halt markers are skipped, mirroring
    the CSV and JUnit exports.
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
            "evidence_used": bool(entry.get("evidence_used")),
            "cached": bool(entry.get("cached")),
        }


def result_to_jsonl(payload: Mapping[str, Any]) -> str:
    """Render per-claim verdicts as newline-delimited JSON objects.

    Each verdict becomes exactly one line, so consumers can stream or
    re-parse partial files without loading the whole session in memory.
    """

    return "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in verdict_records(payload)
    )
