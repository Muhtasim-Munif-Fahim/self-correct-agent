"""Research and policy-writing demo for self-correct-agent.

Shows AntiHallucinator on a research-style policy brief: the mocked draft
mixes a checkable date with a fabricated citation, CoV flags the unsupported
claim, and the rewrite stays conservative.

The mocked OpenAI-compatible client means this runs without API keys.
The notebook version mirrors this script.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

from self_correct import AntiHallucinator, templates

PROMPT_QUESTION = (
    "Did New York City's congestion pricing program reduce Manhattan traffic "
    "after it began in January 2025?"
)

DRAFT = (
    "Yes, official counts show fewer vehicles entering the Manhattan "
    "congestion zone after the program began in January 2025.\n\n"
    "- The program started on 5 January 2025.\n"
    "- Chen et al. (2025) in Nature reported a 47% citywide traffic drop "
    "in the first month.\n\n"
    "If that 47% figure were withdrawn, an immediate citywide expansion "
    "would need to be reconsidered."
)

EXTRACTED_CLAIMS = (
    "1. The congestion pricing program started on 5 January 2025.\n"
    "2. Chen et al. (2025) in Nature reported a 47% citywide traffic drop "
    "in the first month."
)

CORRECTED_BRIEF = (
    "Yes, official counts show fewer vehicles entering the Manhattan "
    "congestion zone after the program began in January 2025.\n\n"
    "- The program started on 5 January 2025.\n\n"
    "The size of the first-month change should be taken from official MTA "
    "or city counts, not from an unsourced 47% citywide figure. If those "
    "official counts were later revised, any expansion recommendation "
    "would need to be revisited."
)


def _response(content: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = content
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    return mock


def build_prompt() -> str:
    """Render the built-in research-answer template for the policy question."""
    body = templates.get_template("research-answer")
    if body is None:
        raise RuntimeError("built-in template 'research-answer' is missing")
    prompt, missing = templates.render(body, {"question": PROMPT_QUESTION})
    if missing or prompt is None:
        raise RuntimeError(f"research-answer template missing values: {missing}")
    return prompt


def build_client() -> MagicMock:
    client = MagicMock()
    responses: List[MagicMock] = [
        _response(DRAFT, 80, 90),
        _response(EXTRACTED_CLAIMS, 70, 40),
        _response("VERIFIED: True.", 30, 8),
        _response(
            "VERIFIED: False. No such Nature paper exists; the 47% citywide "
            "figure is fabricated.",
            35,
            20,
        ),
        _response(CORRECTED_BRIEF, 90, 80),
    ]

    def _create(*args, **kwargs):
        return responses.pop(0)

    client.chat.completions.create.side_effect = _create
    return client


def main() -> None:
    prompt = build_prompt()
    client = build_client()
    agent = AntiHallucinator(
        client=client,
        strictness=1.0,
        draft_system_prompt=(
            "You are a policy analyst. Prefer attributable statistics and "
            "named sources. Do not invent citations."
        ),
    )
    result = agent.generate(model="gpt-4o-mini", prompt=prompt)
    summary = result.claim_summary()

    print("== Research / policy prompt ==")
    print(prompt)
    print()
    print("== Corrected brief ==")
    print(result.content)
    print()
    print("== Flagged claims ==")
    if result.hallucinations_caught:
        for item in result.hallucinations_caught:
            print(f"- {item}")
    else:
        print("(none)")
    print()
    print("== Summary ==")
    print(f"claims checked: {summary['total_claims']}")
    print(f"claims verified: {summary['verified_claims']}")
    print(f"claims flagged: {len(result.hallucinations_caught)}")
    print(f"prompt tokens: {result.token_usage.prompt_tokens}")
    print(f"completion tokens: {result.token_usage.completion_tokens}")
    print(f"total tokens: {result.token_usage.total_tokens}")
    print(f"estimated cost: ${result.token_usage.estimate_cost():.4f}")


if __name__ == "__main__":
    main()
