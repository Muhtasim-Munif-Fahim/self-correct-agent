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
    if "label" in entry:
        try:
            entry["label"] = parse_label(entry["label"])
        except ValueError:
            entry.pop("label", None)
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


def parse_label(raw: object) -> List[str]:
    """Normalize a label value from the CLI or a recorded run.

    Accepts None (returns []), a single string ("alpha" -> ["alpha"]), or an
    iterable of strings (["a", "b"]). Each tag is stripped of surrounding
    whitespace; empty tags raise ``ValueError`` so callers fail fast on
    accidental double commas.
    """

    if raw is None:
        return []
    if isinstance(raw, str):
        candidates = [part for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        candidates = list(raw)
    else:
        raise ValueError("label must be a string or a list of strings")
    tags: List[str] = []
    for item in candidates:
        if not isinstance(item, str):
            raise ValueError("label entries must be strings")
        tag = item.strip()
        if not tag:
            raise ValueError("label entries must be non-empty")
        tags.append(tag)
    return tags


def filter_runs(
    runs: List[Dict[str, Any]],
    *,
    label: Optional[object] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return only the recorded runs matching the requested label/model filters.

    ``label`` accepts the same shapes as :func:`parse_label` (a comma-separated
    string or a list of strings). A run matches when its ``label`` field
    contains every requested tag, so callers can narrow a session to a
    specific cohort without iterating manually. ``model`` matches the recorded
    model string exactly; ``None`` leaves that filter open. Runs whose
    ``label`` field fails to parse are silently treated as unlabeled so an
    old or hand-edited history file cannot break filtering.
    """

    targets = parse_label(label)
    out: List[Dict[str, Any]] = []
    for run in runs:
        if model is not None and str(run.get("model", "")) != model:
            continue
        if targets:
            try:
                run_tags = parse_label(run.get("label"))
            except ValueError:
                continue
            if not all(tag in run_tags for tag in targets):
                continue
        out.append(run)
    return out


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
        "labels": _label_counts(runs),
    }


def _label_counts(runs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count how often each recorded label tag appears across runs."""

    counts: Dict[str, int] = {}
    for run in runs:
        try:
            tags = parse_label(run.get("label"))
        except ValueError:
            continue
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


#: Bucket widths accepted by :func:`trend_buckets`, in seconds.
_BUCKET_SECONDS = {"hour": 3600, "day": 86400, "week": 604800}

#: Metrics where a rising value is a regression rather than an improvement.
LOWER_IS_BETTER = frozenset({"mean_duration", "errors"})


def trend_buckets(
    runs: List[Dict[str, Any]],
    *,
    bucket: str = "day",
    min_runs: int = 1,
) -> List[Dict[str, Any]]:
    """Group runs into fixed time buckets and summarise each one.

    ``aggregate`` collapses the whole history into a single verified rate,
    which hides the thing that actually matters over time: whether
    verification quality is drifting. Bucketing keeps the same statistics but
    leaves the time axis intact.

    Buckets are aligned to the epoch rather than to the first run, so the same
    run always lands in the same bucket no matter which slice of history it is
    computed over. Empty spans between runs produce no bucket. Results are
    ordered oldest first.

    Parameters
    ----------
    bucket:
        One of ``hour``, ``day`` or ``week``.
    min_runs:
        Drop buckets holding fewer than this many runs, which keeps a single
        stray run from reading as a trend. Must be a positive integer.
    """

    if bucket not in _BUCKET_SECONDS:
        raise ValueError(f"bucket must be one of {', '.join(sorted(_BUCKET_SECONDS))}")
    if not isinstance(min_runs, int) or isinstance(min_runs, bool) or min_runs <= 0:
        raise ValueError("min_runs must be a positive integer")

    width = _BUCKET_SECONDS[bucket]
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for run in runs:
        try:
            stamp = float(run.get("timestamp", 0) or 0)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(int(stamp // width) * width, []).append(run)

    out: List[Dict[str, Any]] = []
    for start in sorted(grouped):
        members = grouped[start]
        if len(members) < min_runs:
            continue
        summary = aggregate(members)
        out.append(
            {
                "bucket_start": start,
                "bucket": time.strftime("%Y-%m-%dT%H:%M", time.gmtime(start)),
                "runs": summary["runs"],
                "errors": summary["errors"],
                "claims": summary["claims"],
                "claims_verified": summary["claims_verified"],
                "verified_rate": summary["verified_rate"],
                "mean_duration": summary["mean_duration"],
                "prompt_tokens": summary["prompt_tokens"],
                "completion_tokens": summary["completion_tokens"],
            }
        )
    return out


def trend_direction(
    buckets: List[Dict[str, Any]],
    *,
    metric: str = "verified_rate",
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Compare the first and second half of ``buckets`` on one metric.

    Splits the buckets down the middle, averages ``metric`` over each half and
    reports the change. The verdict is ``improving``, ``declining`` or
    ``stable``; ``tolerance`` is the absolute change below which the two halves
    count as the same, so ordinary noise does not read as a regression.

    The verdict follows the metric's own direction, not the sign of the
    change: a rising ``verified_rate`` is an improvement, but a rising
    ``mean_duration`` or ``errors`` is a regression. ``LOWER_IS_BETTER`` lists
    the metrics that invert. ``change`` is always reported as the raw
    difference so callers can still see which way the number moved.

    With an odd number of buckets the middle one is left out of both halves
    rather than biasing one side. Fewer than two buckets cannot support a
    comparison and return the ``insufficient-data`` verdict.
    """

    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative number")

    values = [float(b[metric]) for b in buckets if metric in b]
    if len(values) < 2:
        return {
            "verdict": "insufficient-data",
            "metric": metric,
            "buckets": len(values),
            "change": 0.0,
        }

    half = len(values) // 2
    earlier = values[:half]
    later = values[-half:]
    first = sum(earlier) / len(earlier)
    second = sum(later) / len(later)
    change = second - first

    # A rising duration or error count is a regression, so the sign of the
    # change alone cannot decide the verdict.
    better = -change if metric in LOWER_IS_BETTER else change
    if abs(change) <= float(tolerance):
        verdict = "stable"
    elif better > 0:
        verdict = "improving"
    else:
        verdict = "declining"

    return {
        "verdict": verdict,
        "metric": metric,
        "buckets": len(values),
        "first_half": first,
        "second_half": second,
        "change": change,
    }
