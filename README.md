# Self-Correct Agent — Chain-of-Verification for Python

[![Tests](https://github.com/Muhtasim-Munif-Fahim/self-correct-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/Muhtasim-Munif-Fahim/self-correct-agent/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue)](https://pypi.org/project/self-correct/)

`self-correct-agent` is a small Python library for wrapping an LLM client with a **Chain-of-Verification (CoV)** workflow. It drafts a response, extracts factual claims, critiques each claim, and rewrites the output when unsupported statements are found.

It is designed for people who want a practical hallucination-reduction layer without having to replace their existing OpenAI-compatible client.

## Integrations

Works with any client that exposes `client.chat.completions.create()`:

| Provider | Setup |
| --- | --- |
| **OpenAI** | `from openai import OpenAI` |
| **Anthropic** (via OpenAI SDK compat) | point `base_url` at your proxy |
| **LiteLLM** | wrap with LiteLLM's OpenAI-compatible interface |
| **Ollama / local** | `OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` |

See [`benchmarks/`](benchmarks/) for a 20-prompt eval harness (mock + live modes).

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
- Pluggable verification **Tool** interface with three built-in backends:
  - `DuckDuckGoSearchTool` — web search (default)
  - `WikipediaSearchTool` — Wikipedia article summaries
  - `StaticKnowledgeTool` — user-provided knowledge base (dict, JSON file, or URL)
- Async claim verification for faster checks on long drafts.
- Thread-safe LRU cache for repeated claim verification.
- Token usage tracking and simple cost estimation.
- Hallucination density scoring: flagged/total claim rate plus per-100-words density on every report.
- Custom prompts for draft, extraction, critique, and correction stages.
- Rich report exports: `to_dict()`, `to_json()`, `to_markdown()`.
- Command-line interface with `verify`, `resume`, `batch`, and `info` subcommands.

## Installation

```bash
pip install self-correct

# Development install with all tools and tests
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
print("density:", response.hallucination_density_report())
print("tokens used:", response.token_usage.total_tokens)
```

## Verification Tools

### DuckDuckGo (web search)

```python
from self_correct import AntiHallucinator, DuckDuckGoSearchTool

safe = AntiHallucinator(
    client=client,
    strictness=1.0,
    tools=[DuckDuckGoSearchTool()],
)

response = safe.generate(model="gpt-4o-mini", prompt="What is the population of Tokyo?")
```

### Wikipedia (article summaries)

```python
from self_correct import AntiHallucinator, WikipediaSearchTool

safe = AntiHallucinator(
    client=client,
    strictness=1.0,
    tools=[WikipediaSearchTool(lang="en")],
)
```

### Static knowledge base

```python
from self_correct import StaticKnowledgeTool

# From a dictionary
kb = StaticKnowledgeTool({
    "tokyo population": "Tokyo has ~14 million residents.",
    "einstein": "Developed the theory of relativity.",
})

# From a JSON file
kb = StaticKnowledgeTool.from_json("knowledge.json")

# From a URL
kb = StaticKnowledgeTool.from_json_url("https://example.com/kb.json")

safe = AntiHallucinator(client=client, tools=[kb])
```

### Multiple tools

```python
safe = AntiHallucinator(
    client=client,
    strictness=1.0,
    tools=[DuckDuckGoSearchTool(), WikipediaSearchTool()],
)
```

## Report Exports

Responses can be exported in multiple formats:

```python
result = safe.generate(model="gpt-4o-mini", prompt="...")

# Plain dictionary
data = result.to_dict()

# JSON string
print(result.to_json(indent=2))

# Markdown report (with optional verification log)
print(result.to_markdown(include_log=True))
```

### Hallucination density

Every result reports how densely flagged claims appear, as both a **claim rate** (flagged ÷ total extracted claims) and an optional **per-100-words** score so short drafts are not unfairly penalised for a single miss. An empty claim log scores `0.0` rather than dividing by zero.

```python
report = result.hallucination_density_report()
# {
#   "total_claims": 4,
#   "flagged_claims": 1,
#   "claim_rate": 0.25,
#   "word_count": 80,
#   "per_words": 100,
#   "per_words_density": 1.25,
# }
```

The same figures are included in `to_dict()` / `to_json()`, the Markdown and HTML reports, and the CLI text output. `VerificationPolicy.max_hallucination_density` gates the per-100-words score.

## CLI

The package ships with a `self-correct` CLI:

```bash
# Verify a single prompt
self-correct verify --model gpt-4o-mini --prompt "Explain quantum computing." --max-tokens 500

# Read prompt from file and output as JSON
self-correct verify --model gpt-4o-mini --file input.txt --output report.json

# Enable verification tools
self-correct verify --model gpt-4o-mini --prompt "..." --tools duckduckgo wikipedia

# Markdown report with full verification log
self-correct verify --model gpt-4o-mini --file input.txt --output-format markdown --include-log

# Save a run, then re-run the same prompt with the saved settings
self-correct verify --model gpt-4o-mini --prompt "Explain transformers." --save-session session.json
self-correct resume session.json

# Batch process multiple prompts (JSONL format)
echo '{"id": "1", "prompt": "Explain transformers"}
{"id": "2", "prompt": "What is RLHF?"}' > prompts.jsonl

self-correct batch --input prompts.jsonl --output results.jsonl --model gpt-4o-mini --format json

# Validate a config file
self-correct config validate --config self-correct.json

# Show package info
self-correct info
```

### Save and resume a verification

`verify --save-session PATH` writes the prompt, settings, and result to a JSON file. `resume` loads that file and re-runs the saved prompt with the saved settings. This is a fresh verification of the same prompt, not a mid-pipeline pause that continues after draft or extract.

```bash
# Save the prompt, settings, and result
self-correct verify --model gpt-4o-mini --prompt "Explain transformers." --save-session session.json

# Re-run the saved prompt with the saved settings
self-correct resume session.json

# Override selected settings for the new run
self-correct resume session.json --model gpt-4o --strictness 0.8 --output report.json

# Persist the new run as its own session
self-correct resume session.json --save-session session-retry.json
```

CLI flags on `resume` replace the matching saved values. The current overrides are `--model`, `--model-draft`, `--model-extract`, `--model-verify`, `--model-correct`, `--strictness`, `--provider`, `--base-url`, `--api-key-env`, `--max-retries`, `--retry-backoff`, `--max-calls`, `--checks`, `--output`, `--output-format`, `--include-log`, `--save-session`, and `--fail-on-hallucination`. Everything else comes from the session file.

Batch `--resume-from` is a different feature: it skips already-completed items in a batch file. Use `resume` for a single saved session.

### Batch JSONL format

Input file (one JSON object per line):

```jsonl
{"id": "001", "prompt": "Explain the Transformer architecture."}
{"id": "002", "prompt": "What is the capital of France?"}
{"id": "003", "prompt": "Describe quantum entanglement."}
```

Output file (adds verification results to each input line):

```jsonl
{"id": "001", "content": "...", "hallucinations_caught": [], "token_usage": {...}, "elapsed_seconds": 1.23}
{"id": "002", "content": "...", "hallucinations_caught": ["Claim '...' flagged: ..."], "token_usage": {...}, "elapsed_seconds": 0.89}
```

## Demo / Examples

The repository includes self-contained examples that use a mocked client, so they run without API keys.

- Script: [`examples/demo.py`](examples/demo.py)
- Notebook: [`examples/demo.ipynb`](examples/demo.ipynb)
- Tool comparison demo: [`examples/tool_comparison_demo.py`](examples/tool_comparison_demo.py)
- Research and policy writing: [`examples/research_policy_demo.py`](examples/research_policy_demo.py) ([notebook](examples/research_policy_demo.ipynb))

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

- [x] ~~Add more reference tools beyond web search.~~ ? v0.2.0
- [x] ~~Expose a small CLI for batch verification workflows.~~ ? v0.2.0
- [x] ~~Add richer reporting formats for verification results.~~ ? v0.2.0
- [x] ~~Publish additional examples for research and policy writing use cases.~~ — [`examples/research_policy_demo.py`](examples/research_policy_demo.py)
- [x] ~~Hallucination density scoring.~~ — flagged/total claim rate plus per-100-words density on responses and reports.
- [ ] Structured output extraction via OpenAI function calling.

## Release Notes

- **v0.2.2** — tools, models, and history subcommands; CSV output format.
- **v0.2.1** — --quiet flag, --verbose flag.
- **v0.2.0** — WikipediaSearchTool, StaticKnowledgeTool, CLI, batch mode, report exports.
- **v0.1.0** — Initial release: CoVe pipeline, DuckDuckGo tool, async, cache.

## References

- Dhuliawala, S. et al. (2023). *Chain-of-Verification Reduces Hallucination in Large Language Models.* [arXiv:2309.11495](https://arxiv.org/abs/2309.11495)
- Min, S. et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* [arXiv:2305.14251](https://arxiv.org/abs/2305.14251)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and pull request guidance.
## Z
## License

MIT - see [LICENSE](LICENSE).
