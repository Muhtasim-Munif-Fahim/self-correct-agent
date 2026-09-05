"""Bundle saved sessions into a single portable archive.

A bundle is one ``tar`` archive (optionally ``gzip``-compressed) that carries
every session file plus a ``manifest.json`` index. The manifest records, per
session, its provenance (model, status), claim counts, severity mix, flag
rate and how many evidence sources were cited, so a bundle can be audited
without extracting and re-parsing each file.

Unlike ``sessions-merge`` (which folds verdicts into one logical session) or
``sessions-export-sqlite`` (which writes one session to a single database),
a bundle preserves every original session untouched and adds an index, so it
suits handoff to a reviewer or long-term audit storage.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from .core import VALID_SEVERITIES, classify_severity

try:
    from . import sessions as _sessions
except Exception:  # pragma: no cover - guarded for import safety
    _sessions = None  # type: ignore[assignment, misc]

#: Version of the bundle manifest schema.
BUNDLE_SCHEMA_VERSION = 1

#: Prefix every archived session file lives under.
SESSIONS_PREFIX = "sessions/"


def _manifest_entry(index: int, path: Path, session: Dict[str, Any]) -> Dict[str, Any]:
    """Build one manifest row for a loaded session."""

    result = session.get("result") or {}
    log = result.get("verification_log") or []
    verdicts = [
        entry
        for entry in log
        if isinstance(entry, dict)
        and "is_valid" in entry
        and entry.get("claim")
    ]
    claims = len(verdicts)
    verified = sum(bool(entry.get("is_valid")) for entry in verdicts)
    severities: Dict[str, int] = {name: 0 for name in VALID_SEVERITIES}
    evidence = 0
    for entry in verdicts:
        if not entry.get("is_valid"):
            severities[classify_severity(str(entry.get("critique", "")))] += 1
        evidence += len(entry.get("evidence_sources") or [])
    return {
        "name": path.name,
        "path": f"{SESSIONS_PREFIX}{index:04d}_{path.name}",
        "model": (session.get("config") or {}).get("model"),
        "status": result.get("status", "unknown"),
        "claims": claims,
        "verified": verified,
        "flagged": claims - verified,
        "flag_rate": (claims - verified) / claims if claims else 0.0,
        "severities": severities,
        "evidence_sources": evidence,
        "modified": path.stat().st_mtime,
    }


def collect_session_files(paths: Iterable[Union[str, Path]]) -> List[Path]:
    """Expand file and directory arguments into saved-session files.

    Delegates to :func:`sessions.collect_session_files` so a bundle honors the
    same directory-scanning and de-duplication rules as the other sessions
    subcommands.
    """

    if _sessions is None:  # pragma: no cover
        raise RuntimeError("sessions module is unavailable")
    return _sessions.collect_session_files(paths)


def create_bundle(
    paths: Iterable[Union[str, Path]],
    output: Union[str, Path],
    *,
    compress: bool = True,
) -> Dict[str, Any]:
    """Bundle saved sessions into a single archive at ``output``.

    Each session file that loads successfully is stored verbatim (re-serialized
    as pretty JSON for a stable, reviewable archive) under
    ``sessions/0001_<name>.json`` and indexed in ``manifest.json``. Files that
    do not load as sessions are listed under ``invalid`` and skipped rather
    than aborting the bundle.

    Returns a summary dict with the archive path, session count, invalid list,
    byte size and whether gzip compression was applied. Raises ``ValueError``
    when no session files are found at all.
    """

    files = collect_session_files(paths)
    if not files:
        raise ValueError("no session files found to bundle")

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if compress else "w"

    entries: List[Dict[str, Any]] = []
    invalid: List[Dict[str, str]] = []
    with tarfile.open(target, mode) as archive:
        for index, path in enumerate(files, 1):
            try:
                session = _sessions.load_session(path)  # type: ignore[union-attr]
            except ValueError as exc:
                invalid.append({"file": str(path), "error": str(exc)})
                continue
            entries.append(_manifest_entry(index, path, session))
            data = json.dumps(session, ensure_ascii=False, indent=2).encode("utf-8")
            member_name = f"{SESSIONS_PREFIX}{index:04d}_{path.name}"
            info = tarfile.TarInfo(name=member_name)
            info.size = len(data)
            info.mtime = time.time()
            archive.addfile(info, io.BytesIO(data))

        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "generated_at": time.time(),
            "generated_by": "self-correct-agent sessions-bundle",
            "total_sessions": len(entries),
            "total_claims": sum(entry["claims"] for entry in entries),
            "total_flagged": sum(entry["flagged"] for entry in entries),
            "sessions": entries,
            "invalid": invalid,
        }
        manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, indent=2
        ).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = time.time()
        archive.addfile(info, io.BytesIO(manifest_bytes))

    return {
        "archive": str(target),
        "sessions": len(entries),
        "invalid": invalid,
        "size_bytes": target.stat().st_size,
        "compressed": compress,
    }


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "SESSIONS_PREFIX",
    "create_bundle",
    "collect_session_files",
]
