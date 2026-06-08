"""Self-contained demo for self-correct-agent.

The demo uses a mocked OpenAI-compatible client so it runs without API keys.
The notebook version mirrors this script.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

from self_correct import AntiHallucinator


def _response(content: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = content
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    return mock


def build_client() -> MagicMock:
    client = MagicMock()
    responses: List[MagicMock] = [
        _response("Moon is cheese. Water boils at 100C.", 40, 20),
        _response("1. Moon is cheese.\n2. Water boils at 100C.", 30, 10),
        _response("VERIFIED: False. The moon is not cheese.", 25, 10),
        _response("VERIFIED: True.", 20, 5),
        _response("Moon is rock. Water boils at 100C.", 45, 20),
    ]

    def _create(*args, **kwargs):
        return responses.pop(0)

    client.chat.completions.create.side_effect = _create
    return client


def main() -> None:
    client = build_client()
    agent = AntiHallucinator(client=client, strictness=1.0)
    result = agent.generate(
        model="gpt-4o-mini",
        prompt="Write two short facts about the moon and water.",
    )

    print("== Corrected content ==")
    print(result.content)
    print()
    print("== Summary ==")
    print(f"claims flagged: {len(result.hallucinations_caught)}")
    print(f"prompt tokens: {result.token_usage.prompt_tokens}")
    print(f"completion tokens: {result.token_usage.completion_tokens}")
    print(f"total tokens: {result.token_usage.total_tokens}")
    print(f"estimated cost: ${result.token_usage.estimate_cost():.4f}")


if __name__ == "__main__":
    main()
