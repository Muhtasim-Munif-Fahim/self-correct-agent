# Self-Correct Agent

[![Tests](https://github.com/Muhtasim-Munif-Fahim/self-correct-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/Muhtasim-Munif-Fahim/self-correct-agent/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`self-correct-agent` is a small Python library for wrapping an LLM client with a Chain-of-Verification workflow. It drafts a response, extracts factual claims, critiques each claim, and rewrites the output when unsupported statements are found.

It is designed for people who want a practical hallucination-reduction layer without having to replace their existing OpenAI-compatible client.

## Problem

LLMs are useful at drafting text, but they still make confident mistakes. In production workflows that matters more than benchmark scores: one fabricated date, one unsupported citation, or one wrong number can make the whole answer unusable.

This package turns that failure mode into a repeatable maintenance step:

1. Draft an answer.
2. Extract discrete factual claims.
3. Verify each claim.
4. Rewrite the draft when claims are weak or false.

## Features

- 4-phase Chain-of-Verification pipeline: draft, extract, critique, correct.
- OpenAI-compatible client support through `client.chat.completions.create()`.
- Optional web search verification with a pluggable `Tool` interface.
- Async claim verification for faster checks on long drafts.
- Thread-safe LRU cache for repeated claim verification.
- Token usage tracking and simple cost estimation.
- Custom prompts for draft, extraction, critique, and correction stages.

## Installation

```bash
pip install self-correct

# Development install with tests and web search support
pip install -e ".[dev,search]"
```

If you prefer pinned local development dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart

```python
from openai import OpenAI
from self_correct import AntiHallucinator

client = OpenAI()
safe = AntiHallucinator(client=client, strictness=1.0)

response = safe.generate(
    model="gpt-4o-mini",
    prompt="Explain the Transformer architecture in two short paragraphs.",
)

print(response.content)
print("claims flagged:", len(response.hallucinations_caught))
print("tokens used:", response.token_usage.total_tokens)
```

## Example With Web Search

```python
from self_correct import AntiHallucinator, DuckDuckGoSearchTool

safe = AntiHallucinator(
    client=client,
    strictness=1.0,
    tools=[DuckDuckGoSearchTool()],
)

response = safe.generate(model="gpt-4o-mini", prompt="What is the population of Tokyo?")
```

## Demo

The repository includes a self-contained demo that uses a mocked client, so it runs without API keys.

- Script: [`examples/demo.py`](examples/demo.py)
- Notebook: [`examples/demo.ipynb`](examples/demo.ipynb)

The screenshot below is a lightweight visual summary of the pipeline and demo output.

![Pipeline demo](assets/demo-screenshot.svg)

## How It Works

1. **Draft** - generate a first-pass response.
2. **Extract** - identify factual claims in the draft.
3. **Critique** - verify each claim, optionally using tools.
4. **Correct** - rewrite the draft to remove unsupported claims.

## API Highlights

```python
safe = AntiHallucinator(
    client=client,
    strictness=1.0,
    cache_size=256,
    draft_system_prompt="You are a careful assistant.",
    extraction_prompt="Extract only factual claims.",
    critique_prompt="Check claims against evidence.",
    correction_prompt="Rewrite conservatively.",
)

print(safe.cache_size)
safe.clear_cache()
```

## Tests

Run the test suite locally:

```bash
python -m pytest -q
```

The CI workflow also runs the demo script so the repository keeps a working example path, not just unit tests.

## Roadmap

- Add richer reporting formats for verification results.
- Add more reference tools beyond web search.
- Expose a small CLI for batch verification workflows.
- Publish additional examples for research and policy writing use cases.

## Release Notes

The repository is maintained with a `v0.1.0` release target and a small, documented public API.

## References

- Dhuliawala, S. et al. (2023). *Chain-of-Verification Reduces Hallucination in Large Language Models.* [arXiv:2309.11495](https://arxiv.org/abs/2309.11495)
- Min, S. et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* [arXiv:2305.14251](https://arxiv.org/abs/2305.14251)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and pull request guidance.

## License

MIT - see [LICENSE](LICENSE).
