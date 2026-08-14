"""Persistent record of verification runs.

The CLI is invoked once per run, so anything that should survive between
invocations — history, aggregate statistics, cache effectiveness — has to be
written somewhere. This module owns that file and nothing else.

Records are JSON Lines: append-only, one self-contained object per run, so a
partially written file still parses up to the last complete line and two
concurrent runs cannot corrupt each other's records.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

#: Environment variable overriding where the history file lives.
HISTORY_PATH_ENV = "SELF_CORRECT_HISTORY"

#: Records kept before the file is trimmed from the front.
MAX_RECORDS = 1000


def history_path() -> Path:
    """Return the history file location.

    Honours SELF_CORRECT_HISTORY so tests and CI can point it somewhere
    disposable instead of the user's home directory.
    """

    override = os.environ.get(HISTORY_PATH_ENV)
    if override:
        return Path(override)
    return Path.home() / ".self-correct" / "history.jsonl"


def record_run(entry: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Append one run to the history file.

    History is a convenience, never the point of the command, so a failure to
    write it must not fail the run the user actually asked for.
    """

    target = path or history_path()
    entry = {"timestamp": time.time(), **entry}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        return
    _trim(target)


def _trim(path: Path) -> None:
    """Keep the file bounded by dropping the oldest records."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        if len(lines) <= MAX_RECORDS:
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.writelines(lines[-MAX_RECORDS:])
    except OSError:
        return


def iter_runs(path: Optional[Path] = None) -> Iterator[Dict[str, Any]]:
    """Yield recorded runs oldest first, skipping any unparseable line."""

    target = path or history_path()
    try:
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A truncated final line from an interrupted write should
                    # not make the whole history unreadable.
                    continue
    except OSError:
        return


def load_runs(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all recorded runs as a list, oldest first."""

    return list(iter_runs(path))


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise a list of runs for the stats subcommand."""

    if not runs:
        return {"runs": 0}

    def _total(key: str) -> int:
        return sum(int(run.get(key) or 0) for run in runs)

    models: Dict[str, int] = {}
    for run in runs:
        model = str(run.get("model", "unknown"))
        models[model] = models.get(model, 0) + 1

    claims = _total("claims")
    verified = _total("claims_verified")
    durations = [float(run["duration"]) for run in runs if run.get("duration") is not None]
    errors = sum(1 for run in runs if run.get("error"))

    return {
        "runs": len(runs),
        "errors": errors,
        "first": min(float(r.get("timestamp", 0)) for r in runs),
        "last": max(float(r.get("timestamp", 0)) for r in runs),
        "claims": claims,
        "claims_verified": verified,
        "verified_rate": (verified / claims) if claims else 0.0,
        "cache_hits": _total("cache_hits"),
        "cache_misses": _total("cache_misses"),
        "prompt_tokens": _total("prompt_tokens"),
        "completion_tokens": _total("completion_tokens"),
        "total_duration": sum(durations),
        "mean_duration": (sum(durations) / len(durations)) if durations else 0.0,
        "models": dict(sorted(models.items(), key=lambda kv: -kv[1])),
    }
