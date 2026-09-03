"""Render a Markdown review summary from a saved session."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def render_session_review(
    session: Mapping[str, Any],
    *,
    top_n: int = 5,
) -> str:
    """Return a human-friendly Markdown review of a saved session.

    The review covers:
      - the session's prompt (truncated to 200 characters) and model config
      - the headline counts (total claims, verified, flagged, hallucination
        rate, critiques-with-content, claims-with-step, status)
      - the top flagged claims by critique length (the longest critiques
        usually contain the most actionable feedback)
      - the distinct checks observed across the verification log
    """
    if not isinstance(session, Mapping):
        raise ValueError("session must be a mapping")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    result = session.get("result") or {}
    log = result.get("verification_log") or []
    flagged = [entry for entry in log if isinstance(entry, dict) and not entry.get("is_valid")]
    verified = [entry for entry in log if isinstance(entry, dict) and entry.get("is_valid")]

    flagged_count = len(flagged)
    verified_count = len(verified)
    total = flagged_count + verified_count
    hallucination_rate = (flagged_count / total) if total else 0.0
    critiques_with_content = sum(
        1 for entry in log
        if isinstance(entry, dict) and str(entry.get("critique") or "").strip()
    )
    with_step = sum(
        1 for entry in log
        if isinstance(entry, dict) and entry.get("step") is not None
    )
    checks = sorted({
        str(entry.get("check") or "").strip()
        for entry in log
        if isinstance(entry, dict) and str(entry.get("check") or "").strip()
    })

    config = session.get("config") or {}
    prompt = str(session.get("prompt", ""))
    truncated_prompt = prompt if len(prompt) <= 200 else prompt[:197] + "..."

    lines: list[str] = ["# Session review", ""]
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Status: `{result.get('status', 'unknown')}`")
    lines.append(f"- Prompt: {truncated_prompt}")
    if config:
        lines.append(f"- Model config: `{_summarise_config(config)}`")
    lines.append("")

    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Total claims | {total} |")
    lines.append(f"| Verified | {verified_count} |")
    lines.append(f"| Flagged | {flagged_count} |")
    lines.append(f"| Hallucination rate | {hallucination_rate:.2%} |")
    lines.append(f"| Critiques with content | {critiques_with_content} |")
    lines.append(f"| Claims with step | {with_step} |")
    lines.append(f"| Distinct checks | {len(checks)} |")
    lines.append("")

    top_flagged = sorted(
        flagged,
        key=lambda entry: len(str(entry.get("critique") or "")),
        reverse=True,
    )[:top_n]
    lines.append(f"## Top {min(top_n, len(top_flagged))} flagged claims")
    lines.append("")
    if not top_flagged:
        lines.append("No flagged claims in this session.")
    else:
        for idx, entry in enumerate(top_flagged, start=1):
            claim = str(entry.get("claim", ""))
            critique = str(entry.get("critique") or "").strip() or "(no critique)"
            lines.append(f"{idx}. **{claim}**")
            lines.append(f"   - Critique: {critique}")
    lines.append("")

    if checks:
        lines.append("## Distinct checks")
        lines.append("")
        for check in checks:
            lines.append(f"- `{check}`")
        lines.append("")

    return "\n".join(lines)


def render_session_review_with_counts(
    session: Mapping[str, Any],
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """Return a JSON-friendly summary alongside the Markdown body.

    Useful for callers that want to wire the review into a dashboard and
    need both the human-readable text and the structured counters.
    """

    log = (session.get("result") or {}).get("verification_log") or []
    flagged = [entry for entry in log if isinstance(entry, dict) and not entry.get("is_valid")]
    verified = [entry for entry in log if isinstance(entry, dict) and entry.get("is_valid")]
    return {
        "markdown": render_session_review(session, top_n=top_n),
        "counts": {
            "total_claims": len(verified) + len(flagged),
            "verified": len(verified),
            "flagged": len(flagged),
            "checks_seen": sorted({
                str(entry.get("check") or "").strip()
                for entry in log
                if isinstance(entry, dict) and str(entry.get("check") or "").strip()
            }),
        },
    }


def _summarise_config(config: Mapping[str, Any]) -> str:
    if not config:
        return "(empty)"
    keys = sorted(config.keys())
    return "{ " + ", ".join(keys) + " }"


__all__ = ["render_session_review", "render_session_review_with_counts"]
