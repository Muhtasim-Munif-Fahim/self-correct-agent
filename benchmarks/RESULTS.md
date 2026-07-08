# Example benchmark results

Replace this template after a live run with your model and tool configuration.

| Model | Mode | Prompts | Claims flagged (avg) | Rewrite rate | Notes |
| --- | --- | --- | --- | --- | --- |
| gpt-4o-mini | 1-pass | 20 | — | — | Baseline draft only |
| gpt-4o-mini | CoV + DuckDuckGo | 20 | TBD | TBD | Run `python run_benchmark.py --live` |
| gpt-4o-mini | CoV + StaticKnowledgeTool | 20 | TBD | TBD | Optional ablation |

**How to update:** copy the table from `run_benchmark.py` stdout after a live run, or export JSON with `--output results.json`.
