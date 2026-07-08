# Chain-of-Verification benchmark prompts

Twenty short factual prompts for comparing **single-pass** vs **CoV (self-correct)** outputs.

## Files

| File | Purpose |
| --- | --- |
| [`prompts.jsonl`](prompts.jsonl) | Evaluation prompts (id, category, notes) |
| [`run_benchmark.py`](run_benchmark.py) | Mock-client harness — runs offline in CI |
| [`RESULTS.md`](RESULTS.md) | Example scorecard template |

## Quick run (no API key)

```bash
cd benchmarks
python run_benchmark.py
```

Uses a mocked OpenAI-compatible client so the harness validates the pipeline without network calls.

## With a real model

```bash
export OPENAI_API_KEY=sk-...
python run_benchmark.py --live --model gpt-4o-mini --limit 5
```

## What to measure

| Metric | Definition |
| --- | --- |
| Claims extracted | Count of atomic factual claims per response |
| Claims flagged | Claims marked unsupported after verification |
| Rewrite rate | Fraction of prompts where the final draft differs from pass 1 |
| Latency | Wall-clock seconds (CoV adds rounds) |

Record results in `RESULTS.md` when you run a live eval — even a 20-prompt table builds trust for adopters.

## Citation

If you use this benchmark in a paper or blog post, link to the repo:

https://github.com/Muhtasim-Munif-Fahim/self-correct-agent/tree/main/benchmarks
