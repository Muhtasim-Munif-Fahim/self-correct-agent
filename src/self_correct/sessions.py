"""Portable verification sessions for pausing and resuming CLI work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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
