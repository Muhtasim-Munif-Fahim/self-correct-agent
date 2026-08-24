"""Portable verification sessions for pausing and resuming CLI work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

SESSION_SCHEMA_VERSION = 1


def save_session(
    path: str | Path,
    *,
    prompt: str,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Path:
    """Write a complete, versioned verification session to ``path``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "prompt": prompt,
        "config": dict(config),
        "result": dict(result),
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def load_session(path: str | Path) -> dict[str, Any]:
    """Load and validate a session created by :func:`save_session`."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read session '{source}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"session '{source}' is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("session must contain a JSON object")
    if payload.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported session schema version: "
            f"{payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
        raise ValueError("session prompt must be a non-empty string")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("session config must be a JSON object")
    if not isinstance(payload.get("result"), dict):
        raise ValueError("session result must be a JSON object")
    return payload


def _verdict_map(payload: Mapping[str, Any]) -> Dict[str, bool]:
    """Map normalized claim text to its verdict from a session or result."""

    result = payload.get("result")
    if not isinstance(result, dict):
        result = payload
    verdicts: Dict[str, bool] = {}
    for entry in result.get("verification_log") or []:
        if not isinstance(entry, dict):
            continue
        if "is_valid" not in entry or not entry.get("claim"):
            continue
        verdicts[str(entry["claim"]).strip().lower()] = bool(entry["is_valid"])
    return verdicts


def diff_sessions(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare per-claim verdicts between two saved verification runs.

    Claims are matched by normalized text, mirroring the claim-cache keying,
    so reworded whitespace or casing does not hide an unchanged claim.
    Accepts full sessions (as written by :func:`save_session`) or the bare
    result objects they embed.
    """

    before = _verdict_map(baseline)
    after = _verdict_map(current)
    shared = set(before) & set(after)
    return {
        "resolved": sorted(c for c in shared if not before[c] and after[c]),
        "regressed": sorted(c for c in shared if before[c] and not after[c]),
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "unchanged_verified": sum(1 for c in shared if before[c] and after[c]),
        "unchanged_flagged": sum(1 for c in shared if not before[c] and not after[c]),
        "baseline_claims": len(before),
        "current_claims": len(after),
    }
