# Self-Correct Agent

[![Tests](https://github.com/Muhtasim-Munif-Fahim/self-correct-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/Muhtasim-Munif-Fahim/self-correct-agent/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A lightweight Python library that **automatically detects and prevents LLM hallucinations** using the Chain-of-Verification (CoVe) methodology.

## Why?

LLMs hallucinate. They invent paper titles, fabricate statistics, and confidently state false facts. While models are bad at generating truth on the first try, research shows they are surprisingly good at **catching their own mistakes** when asked to self-critique ([Dhuliawala et al., 2023](https://arxiv.org/abs/2309.11495)).

`self-correct-agent` packages this research into a **drop-in wrapper** around your existing LLM client.

## Features

- 🔄 **4-Phase CoVe Pipeline** — Draft → Extract Claims → Critique → Correct
- 🔍 **Web Search Verification** — Pluggable tools (DuckDuckGo included) for grounding claims in real-world evidence
- ⚡ **Async Parallel Verification** — `generate_async()` verifies all claims concurrently
- 💰 **Token & Cost Tracking** — Full visibility into how many tokens verification costs
- 🧠 **LRU Claim Cache** — Repeated claims skip re-verification, saving time and money
- 🎚️ **Dynamic Strictness** — From `0.0` (passthrough) to `1.0` (strict empirical + tools)

## Installation

```bash
pip install self-correct

# With web search support:
pip install self-correct[search]
```

## Quick Start

```python
import openai
from self_correct import AntiHallucinator

client = openai.OpenAI()
safe = AntiHallucinator(client=client, strictness=1.0)

response = safe.generate(
    model="gpt-4o",
    prompt="Explain the Transformer architecture from Vaswani et al. (2017)."
)

print(response.content)
print(f"Hallucinations caught: {len(response.hallucinations_caught)}")
print(f"Tokens used: {response.token_usage.total_tokens}")
print(f"Estimated cost: ${response.token_usage.estimate_cost():.4f}")
print(f"Time: {response.elapsed_seconds:.1f}s")
```

## Web Search Verification

At `strictness >= 0.8`, provide tools so the verifier searches the web before judging each claim:

```python
from self_correct import AntiHallucinator, DuckDuckGoSearchTool

safe = AntiHallucinator(
    client=client,
    strictness=1.0,
    tools=[DuckDuckGoSearchTool()],
)

response = safe.generate(model="gpt-4o", prompt="Population of Tokyo?")

for entry in response.verification_log:
    print(f"  Claim: {entry['claim']}")
    print(f"  Evidence used: {entry['evidence_used']}")
    print(f"  Valid: {entry['is_valid']}")
    print(f"  Cached: {entry['cached']}")
```

## Async Parallel Verification

For faster verification when many claims are extracted:

```python
import asyncio
from self_correct import AntiHallucinator

safe = AntiHallucinator(client=client, strictness=1.0)
result = asyncio.run(safe.generate_async(model="gpt-4o", prompt="History of Python"))
```

## Custom Tools

Implement the `Tool` interface to add any verification backend:

```python
from self_correct import Tool, SearchResult

class GoogleScholarTool(Tool):
    @property
    def name(self) -> str:
        return "Google Scholar"

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        # Your API logic here
        return [...]

safe = AntiHallucinator(client=client, tools=[GoogleScholarTool()])
```

## Dynamic Strictness

| Level | Behavior |
|-------|----------|
| `0.0` | Passthrough — no verification |
| `0.5` | Light logical critique (catches obvious errors) |
| `0.8` | Strict critique + web search tools (if provided) |
| `1.0` | Maximum strictness — removes unverifiable claims |

## How It Works

```
User Prompt
    │
    ▼
┌─────────────┐
│  1. DRAFT   │  → Generate initial response
└──────┬──────┘
       │
┌──────▼──────┐
│ 2. EXTRACT  │  → Parse all factual claims
└──────┬──────┘
       │
┌──────▼──────┐
│ 3. CRITIQUE │  → Verify each claim (+ web search)
│   [parallel]│    Cache results for reuse
└──────┬──────┘
       │
┌──────▼──────┐
│ 4. CORRECT  │  → Rewrite draft, removing false claims
└──────┬──────┘
       │
       ▼
  Verified Output
```

## References

- Dhuliawala, S. et al. (2023). *Chain-of-Verification Reduces Hallucination in Large Language Models.* [arXiv:2309.11495](https://arxiv.org/abs/2309.11495)
- Min, S. et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision.* [arXiv:2305.14251](https://arxiv.org/abs/2305.14251)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE).
