#!/usr/bin/env python3
"""Offline-friendly benchmark harness for self-correct-agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROMPTS_PATH = Path(__file__).with_name("prompts.jsonl")


def load_prompts(limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


class _MockCompletions:
    def create(self, **kwargs: Any) -> SimpleNamespace:
        messages = kwargs.get("messages", [])
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        if "extract" in user.lower() or "claims" in user.lower():
            content = '["Mock factual claim about the prompt."]'
        elif "critique" in user.lower() or "verify" in user.lower():
            content = "SUPPORTED: mock evidence found."
        elif "rewrite" in user.lower() or "correct" in user.lower():
            content = "Corrected mock answer with verified details."
        else:
            content = f"Draft answer for: {user[:80]}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


class MockClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_MockCompletions())


def run_mock(limit: int | None) -> dict[str, Any]:
    from self_correct import AntiHallucinator

    client = MockClient()
    safe = AntiHallucinator(client=client, strictness=0.8)
    prompts = load_prompts(limit)

    flagged_total = 0
    rewritten = 0
    for row in prompts:
        response = safe.generate(model="mock", prompt=row["prompt"])
        flagged_total += len(response.hallucinations_caught)
        if response.content and "Corrected" in response.content:
            rewritten += 1

    n = len(prompts) or 1
    return {
        "mode": "mock",
        "prompts": len(prompts),
        "avg_claims_flagged": round(flagged_total / n, 2),
        "rewrite_rate": round(rewritten / n, 2),
    }


def run_live(model: str, limit: int | None) -> dict[str, Any]:
    from openai import OpenAI
    from self_correct import AntiHallucinator

    client = OpenAI()
    safe = AntiHallucinator(client=client, strictness=1.0)
    prompts = load_prompts(limit)

    flagged_total = 0
    for row in prompts:
        response = safe.generate(model=model, prompt=row["prompt"])
        flagged_total += len(response.hallucinations_caught)

    n = len(prompts) or 1
    return {
        "mode": "live",
        "model": model,
        "prompts": len(prompts),
        "avg_claims_flagged": round(flagged_total / n, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoV benchmark prompts")
    parser.add_argument("--live", action="store_true", help="Use a real OpenAI client")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = run_live(args.model, args.limit) if args.live else run_mock(args.limit)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
