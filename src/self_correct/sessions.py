"""Portable verification sessions for pausing and resuming CLI work."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .core import VALID_SEVERITIES, classify_severity

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

def collect_session_files(paths: Iterable[str | Path]) -> List[Path]:
    """Expand file and directory arguments into saved-session files.

    A directory contributes its ``*.json`` entries in name order; anything
    else is taken as one file path. Duplicates are dropped so overlapping
    arguments do not count a session twice.
    """

    files: List[Path] = []
    seen = set()
    for entry in paths:
        candidate = Path(entry)
        matches = sorted(candidate.glob("*.json")) if candidate.is_dir() else [candidate]
        for match in matches:
            key = str(match)
            if key not in seen:
                seen.add(key)
                files.append(match)
    return files


def summarize_session_file(
    path: str | Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Summarise claim outcomes in one saved session.

    Returns ``(summary, None)`` on success and ``(None, reason)`` when the
    file cannot be read or is not a valid session, so a scan over many files
    can skip broken ones without losing the reason why.
    """

    try:
        session = load_session(path)
    except ValueError as exc:
        return None, str(exc)

    verdicts = [
        entry
        for entry in session["result"].get("verification_log") or []
        if isinstance(entry, dict) and "is_valid" in entry
    ]
    verified = sum(bool(entry["is_valid"]) for entry in verdicts)
    claims = len(verdicts)
    severities = {name: 0 for name in VALID_SEVERITIES}
    for entry in verdicts:
        if entry["is_valid"] is False:
            severities[classify_severity(str(entry.get("critique", "")))] += 1
    summary = {
        "file": str(path),
        "claims": claims,
        "verified": verified,
        "flagged": claims - verified,
        "flag_rate": (claims - verified) / claims if claims else 0.0,
        "severities": severities,
        "modified": Path(path).stat().st_mtime,
    }
    return summary, None


def aggregate_sessions(paths: Iterable[str | Path]) -> Dict[str, Any]:
    """Aggregate claim analytics across many saved session files.

    The result carries per-session rows sorted oldest to newest by file
    modification time (the trend view), totals across every valid session,
    and an ``invalid`` list describing files that could not be read.
    """

    summaries: List[Dict[str, Any]] = []
    invalid: List[Dict[str, str]] = []
    for path in collect_session_files(paths):
        summary, error = summarize_session_file(path)
        if error is not None:
            invalid.append({"file": str(path), "error": error})
        else:
            summaries.append(summary)
    summaries.sort(key=lambda item: item["modified"])

    claims = sum(item["claims"] for item in summaries)
    verified = sum(item["verified"] for item in summaries)
    flagged = claims - verified
    totals: Dict[str, Any] = {
        "sessions": len(summaries),
        "invalid_files": len(invalid),
        "claims": claims,
        "verified": verified,
        "flagged": flagged,
        "flag_rate": flagged / claims if claims else 0.0,
        "severities": {
            name: sum(item["severities"][name] for item in summaries)
            for name in VALID_SEVERITIES
        },
    }
    return {"sessions": summaries, "invalid": invalid, "totals": totals}


def prune_sessions(
    paths: Iterable[str | Path],
    older_than_days: float,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """List or delete saved sessions older than a cutoff age.

    Only files that load as valid sessions are ever touched; anything
    modified within the last ``older_than_days`` days is kept and counted,
    so fresh or in-progress sessions are never candidates. Files that do not
    parse as sessions are reported as invalid and never deleted. In
    ``dry_run`` mode nothing is removed and the matching files are returned
    under ``candidates`` instead of ``deleted``, so the destructive call can
    be previewed first.
    """

    cutoff = time.time() - older_than_days * 86400
    candidates: List[str] = []
    kept: List[str] = []
    invalid: List[Dict[str, str]] = []
    for path in collect_session_files(paths):
        try:
            load_session(path)
        except ValueError as exc:
            invalid.append({"file": str(path), "error": str(exc)})
            continue
        if Path(path).stat().st_mtime >= cutoff:
            kept.append(str(path))
        else:
            candidates.append(str(path))

    deleted: List[str] = []
    if not dry_run:
        for candidate in candidates:
            try:
                Path(candidate).unlink()
                deleted.append(candidate)
            except OSError as exc:
                invalid.append({"file": candidate, "error": str(exc)})

    result: Dict[str, Any] = {
        "dry_run": bool(dry_run),
        "older_than_days": older_than_days,
        "kept": kept,
        "invalid": invalid,
    }
    result["deleted" if not dry_run else "candidates"] = (
        deleted if not dry_run else candidates
    )
    return result


def search_sessions(
    paths: Iterable[str | Path],
    *,
    claim_query: Optional[str] = None,
    critique_query: Optional[str] = None,
    verdict: Optional[str] = None,
) -> Dict[str, Any]:
    """Find claim verdicts across saved sessions matching text filters.

    Text filters are case-insensitive substring checks: ``claim_query``
    applies to the claim text and ``critique_query`` to its critique.
    ``verdict`` narrows results to ``"verified"`` or ``"flagged"`` claims;
    every filter that is set must match (logical AND). The result carries
    per-match rows with the source file, claim, verdict and critique, how
    many sessions were scanned, and which files were skipped as invalid.
    """

    claim_lookup = claim_query.lower() if claim_query else None
    critique_lookup = critique_query.lower() if critique_query else None

    matches: List[Dict[str, Any]] = []
    scanned = 0
    invalid: List[Dict[str, str]] = []
    for path in collect_session_files(paths):
        try:
            session = load_session(path)
        except ValueError as exc:
            invalid.append({"file": str(path), "error": str(exc)})
            continue
        scanned += 1
        for idx, entry in enumerate(session["result"].get("verification_log") or [], 1):
            if not isinstance(entry, dict):
                continue
            if "is_valid" not in entry or not entry.get("claim"):
                continue
            claim_text = str(entry["claim"])
            is_valid = bool(entry["is_valid"])
            critique = str(entry.get("critique", ""))
            if claim_lookup and claim_lookup not in claim_text.lower():
                continue
            if critique_lookup and critique_lookup not in critique.lower():
                continue
            if verdict == "verified" and not is_valid:
                continue
            if verdict == "flagged" and is_valid:
                continue
            matches.append({
                "file": str(path),
                "id": idx,
                "claim": claim_text,
                "verdict": "verified" if is_valid else "flagged",
                "critique": critique,
            })
    return {
        "match_count": len(matches),
        "scanned": scanned,
        "invalid": invalid,
        "matches": matches,
    }


def _claim_key(entry: Mapping[str, Any]) -> str:
    claim = str(entry.get("claim", "")).strip().lower()
    return claim


def merge_sessions(
    paths: Iterable[str | Path],
    *,
    keep: str = "flag",
    on_conflict: str = "skip",
) -> Dict[str, Any]:
    """Combine multiple saved sessions into a single logical result.

    Sessions are loaded from ``paths`` (files or directories via
    :func:`collect_session_files`); each entry that is not a valid session is
    listed under ``invalid`` and skipped. The merged verification log keeps
    one row per normalized claim text, with conflicts resolved by ``keep``
    (``"flag"`` keeps the flagged verdict, ``"verify"`` keeps the verified
    verdict, ``"any"`` keeps the first verdict seen). When two sessions
    disagree on the verdict of the same claim, ``on_conflict`` is applied:
    ``"skip"`` removes the disputed claim entirely, ``"keep"`` keeps the
    winning row only (the loser is dropped) so the disagreement is resolved
    rather than duplicated.

    The returned object mirrors :func:`save_session`'s shape so the consumer
    can persist it directly with :func:`save_session`. ``source_files`` and
    ``conflicts`` are diagnostic metadata; ``result.hallucinations_caught``
    is recomputed from the merged log so downstream flags stay in sync.
    """
    if keep not in {"flag", "verify", "any"}:
        raise ValueError("keep must be 'flag', 'verify', or 'any'")
    if on_conflict not in {"skip", "keep"}:
        raise ValueError("on_conflict must be 'skip' or 'keep'")

    invalid: List[Dict[str, str]] = []
    source_files: List[str] = []
    seen_keys: Dict[str, Dict[str, Any]] = {}
    conflicts: List[Dict[str, Any]] = []

    for path in collect_session_files(paths):
        try:
            session = load_session(path)
        except ValueError as exc:
            invalid.append({"file": str(path), "error": str(exc)})
            continue
        source_files.append(str(path))
        for entry in session.get("result", {}).get("verification_log") or []:
            if not isinstance(entry, dict):
                continue
            if "is_valid" not in entry:
                continue
            claim_text = str(entry.get("claim", "")).strip()
            if not claim_text:
                continue
            key = _claim_key(entry)
            is_valid = bool(entry.get("is_valid"))
            existing = seen_keys.get(key)
            if existing is None:
                seen_keys[key] = {
                    "entry": dict(entry),
                    "source": str(path),
                    "verdicts": [is_valid],
                }
                continue
            existing["verdicts"].append(is_valid)
            existing_was_flagged = not bool(existing["entry"].get("is_valid"))
            new_is_flagged = not is_valid
            if existing_was_flagged == new_is_flagged:
                # Same verdict: keep the first one seen.
                continue
            conflicts.append(
                {
                    "claim": str(entry.get("claim", "")),
                    "verdicts": list(existing["verdicts"]),
                    "sources": sorted({existing["source"], str(path)}),
                }
            )
            winner_is_flag = (
                (keep == "flag" and is_valid is False)
                or (keep == "verify" and is_valid is True)
            )
            if keep == "any":
                # First verdict seen wins; drop the newcomer.
                continue
            existing_wins = (
                (keep == "flag" and existing_was_flagged)
                or (keep == "verify" and not existing_was_flagged)
            )
            if existing_wins:
                continue
            existing["entry"] = dict(entry)
            existing["source"] = str(path)

    merged_log: List[Dict[str, Any]] = []
    for key, info in seen_keys.items():
        if on_conflict == "skip" and len(info["verdicts"]) > 1 and (
            not all(v == info["verdicts"][0] for v in info["verdicts"])
        ):
            continue
        merged_log.append(info["entry"])

    flagged = [str(entry.get("claim", "")) for entry in merged_log if not entry.get("is_valid")]
    hallucination_set: List[str] = []
    for claim in flagged:
        if claim and claim not in hallucination_set:
            hallucination_set.append(claim)

    result: Dict[str, Any] = {
        "status": (
            "flagged" if hallucination_set else (
                "verified" if merged_log else "unknown"
            )
        ),
        "content": "",
        "verification_log": merged_log,
        "hallucinations_caught": hallucination_set,
    }
    return {
        "source_files": source_files,
        "invalid": invalid,
        "conflicts": conflicts,
        "result": result,
    }


def export_to_sqlite(
    sqlite_path: str | Path,
    session: Mapping[str, Any],
    *,
    table_name: str = "verification_log",
) -> int:
    """Write the session's verification log to a SQLite database.

    Two tables are created:
      - ``{table_name}``: one row per (claim, critique, verdict) record;
      - ``{table_name}_meta``: one row summarising the session provenance
        (session_id, prompt, model, status, source_file).

    The function is dependency-free (only the standard library
    ``sqlite3`` module is used). Returns the number of rows written to
    the verification log table. An existing database at ``sqlite_path`` is
    replaced. ``table_name`` must be a safe SQL identifier; ``ValueError``
    is raised for anything else.
    """

    import sqlite3

    if not isinstance(table_name, str) or not table_name:
        raise ValueError("table_name must be a non-empty string")
    if not table_name.replace("_", "").isalnum():
        raise ValueError("table_name must be alphanumeric with underscores")
    log_table = table_name
    meta_table = f"{table_name}_meta"

    path = Path(sqlite_path)
    if path.parent and path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            # On Windows the file may be in use; truncate instead.
            with open(path, "w", encoding="utf-8") as handle:
                handle.truncate(0)
    with sqlite3.connect(path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            f"CREATE TABLE {log_table} ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "claim TEXT, "
            "is_valid INTEGER, "
            "critique TEXT, "
            "step INTEGER, "
            "timestamp TEXT"
            ")"
        )
        cursor.execute(
            f"CREATE TABLE {meta_table} ("
            "id INTEGER PRIMARY KEY, "
            "session_prompt TEXT, "
            "config_json TEXT, "
            "status TEXT, "
            "source_file TEXT"
            ")"
        )
        rows = 0
        for entry in session.get("result", {}).get("verification_log") or []:
            if not isinstance(entry, dict):
                continue
            cursor.execute(
                f"INSERT INTO {log_table} (claim, is_valid, critique, step, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(entry.get("claim", "")),
                    int(bool(entry.get("is_valid", False))),
                    str(entry.get("critique", "")),
                    entry.get("step"),
                    str(entry.get("timestamp", "")),
                ),
            )
            rows += 1
        cursor.execute(
            f"INSERT INTO {meta_table} "
            "(session_prompt, config_json, status, source_file) "
            "VALUES (?, ?, ?, ?)",
            (
                str(session.get("prompt", "")),
                json.dumps(session.get("config", {}), default=str),
                str((session.get("result") or {}).get("status", "")),
                str(path.with_suffix(".json")),
            ),
        )
        connection.commit()
    return rows