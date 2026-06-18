"""Command-line interface for self-correct-agent.

Provides batch verification workflows, file-based prompts,
and configurable output formats.

Usage::

    self-correct verify --model gpt-4o-mini --prompt "Explain quantum computing."
    self-correct verify --model gpt-4o-mini --file input.txt --output report.json
    self-correct verify --model gpt-4o-mini --file batch.txt --output-format markdown
"""

import argparse
import json
import sys
import time
from typing import Optional

from .core import AntiHallucinator
from .tools import DuckDuckGoSearchTool, WikipediaSearchTool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self-correct",
        description="Anti-hallucination wrapper for LLMs using Chain-of-Verification.",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and exit",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--quiet", action="store_true", help="Suppress all non-essential output")
    sub = parser.add_subparsers(dest="command", required=True)

    # verify subcommand
    verify = sub.add_parser("verify", help="Generate and verify text")
    verify.add_argument(
        "--model", default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)",
    )
    verify.add_argument(
        "--prompt", "-p", default=None,
        help="Prompt text to process",
    )
    verify.add_argument(
        "--file", "-f", default=None,
        help="Read prompt from file instead of --prompt",
    )
    verify.add_argument(
        "--strictness", type=float, default=1.0,
        help="Verification strictness 0.0-1.0 (default: 1.0)",
    )
    verify.add_argument(
        "--tools", nargs="*", default=[],
        choices=["duckduckgo", "wikipedia"],
        help="Verification tools to enable",
    )
    verify.add_argument(
        "--output", "-o", default=None,
        help="Write output to file (detects format from extension: .json, .md, .txt)",
    )
    verify.add_argument(
        "--output-format", default=None,
        choices=["json", "markdown", "text", "csv"],
        help="Output format (overrides auto-detection from --output)",
    )
    verify.add_argument(
        "--include-log", action="store_true",
        help="Include full verification log in markdown output",
    )
    verify.add_argument(
        "--no-cache", action="store_true",
        help="Disable claim verification cache",
    )

    # tools subcommand
    tools_parser = sub.add_parser("tools", help="List available verification tools")

    # models subcommand
    models_parser = sub.add_parser("models", help="List supported models with estimated costs")

    # history subcommand
    history_parser = sub.add_parser("history", help="Show recent verification history (current session)")

    # info subcommand
    info = sub.add_parser("info", help="Show package information")

    # batch subcommand
    batch = sub.add_parser("batch", help="Process multiple prompts from a JSONL file")
    batch.add_argument(
        "--input", "-i", required=True,
        help="Input JSONL file (each line: {\"id\": \"...\", \"prompt\": \"...\"})",
    )
    batch.add_argument(
        "--output", "-o", default=None,
        help="Output JSONL file (default: stdout)",
    )
    batch.add_argument(
        "--model", default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)",
    )
    batch.add_argument(
        "--strictness", type=float, default=1.0,
        help="Verification strictness 0.0-1.0 (default: 1.0)",
    )
    batch.add_argument(
        "--tools", nargs="*", default=[],
        choices=["duckduckgo", "wikipedia"],
        help="Verification tools to enable",
    )
    batch.add_argument(
        "--max-items", type=int, default=None,
        help="Maximum number of items to process (default: all)",
    )
    batch.add_argument(
        "--no-cache", action="store_true",
        help="Disable claim verification cache",
    )
    batch.add_argument(
        "--delay", type=float, default=0.0,
        help="Delay in seconds between items (to avoid rate limits)",
    )

    return parser


def _read_prompt(prompt: Optional[str], file_path: Optional[str]) -> str:
    """Read the prompt from string argument or file."""
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    if prompt:
        return prompt
    raise ValueError("Either --prompt or --file must be provided.")


def _detect_output_format(output_path: Optional[str], format_override: Optional[str]) -> str:
    """Detect output format from file extension or explicit argument."""
    if format_override:
        return format_override
    if output_path:
        if output_path.endswith(".json"):
            return "json"
        if output_path.endswith(".md"):
            return "markdown"
    return "text"


def cmd_verify(args: argparse.Namespace) -> None:
    """Execute the verify subcommand."""
    from openai import OpenAI

    prompt = _read_prompt(args.prompt, args.file)

    client = OpenAI()

    tools = []
    if "duckduckgo" in args.tools:
        tools.append(DuckDuckGoSearchTool())
    if "wikipedia" in args.tools:
        tools.append(WikipediaSearchTool())

    hallu = AntiHallucinator(
        client=client,
        strictness=args.strictness,
        tools=tools or None,
        cache_size=0 if args.no_cache else 256,
    )

    result = hallu.generate(model=args.model, prompt=prompt)

    output_format = _detect_output_format(args.output, args.output_format)

    if output_format == "json":
        output = result.to_json()
    elif output_format == "markdown":
        output = result.to_markdown(include_log=args.include_log)
    elif output_format == "csv":
        import csv, io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["status", "tokens", "hallucinations", "output"])
        w.writerow([result.status, result.token_usage.total_tokens, len(result.hallucinations_caught), result.content[:200]])
        output = buf.getvalue()
    else:
        # text format
        lines = [
            "=" * 60,
            "SELF-CORRECT AGENT - VERIFICATION REPORT",
            "=" * 60,
            "",
            f"Tokens: {result.token_usage.total_tokens}",
            f"Duration: {result.elapsed_seconds:.2f}s",
            f"Hallucinations caught: {len(result.hallucinations_caught)}",
            "",
        ]
        if result.hallucinations_caught:
            lines.append("--- Flagged Claims ---")
            for h in result.hallucinations_caught:
                lines.append(f"   {h}")
            lines.append("")
        lines.append("--- Final Output ---")
        lines.append(result.content)
        lines.append("")
        lines.append("=" * 60)
        output = "\n".join(lines)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    if not getattr(args, "quiet", False):
        if args.verbose:
            print(f"Verbose: {result.token_usage.total_tokens} tokens, {result.elapsed_seconds:.2f}s", file=sys.stderr)


def cmd_info() -> None:
    """Show package information."""
    from . import __version__
    info = {
        "package": "self-correct",
        "version": __version__,
        "description": "A lightweight anti-hallucination wrapper for LLMs using Chain-of-Verification.",
    }
    print(json.dumps(info, indent=2))


def _cmd_history() -> int:
    print("History tracking is not yet implemented.")
    return 0


def _cmd_tools() -> int:
    tools = [
        ("DuckDuckGoSearchTool", "Web search via DuckDuckGo (no API key needed)"),
        ("WikipediaSearchTool", "Wikipedia article summaries (requires wikipedia package)"),
        ("StaticKnowledgeTool", "User-provided knowledge base (dict, JSON file URL)"),
    ]
    print(f"{'Tool':<25} {'Description':<60}")
    print("-" * 85)
    for name, desc in tools:
        print(f"{name:<25} {desc:<60}")
    return 0


def _cmd_models() -> int:
    models = [
        ("gpt-4o-mini", "$0.15", "$0.60"),
        ("gpt-4o", "$2.50", "$10.00"),
        ("gpt-4-turbo", "$10.00", "$30.00"),
        ("gpt-3.5-turbo", "$0.50", "$1.50"),
    ]
    print(f"{'Model':<25} {'Input/1M':<15} {'Output/1M':<15}")
    print("-" * 55)
    for name, inp, out in models:
        print(f"{name:<25} {inp:<15} {out:<15}")
    print("\n* Prices per 1M tokens, approximate. Check provider for current pricing.")
    return 0


def cmd_batch(args: argparse.Namespace) -> None:
    """Execute the batch subcommand: process a JSONL file."""
    from openai import OpenAI

    tools = []
    if "duckduckgo" in args.tools:
        tools.append(DuckDuckGoSearchTool())
    if "wikipedia" in args.tools:
        tools.append(WikipediaSearchTool())

    hallu = AntiHallucinator(
        client=OpenAI(),
        strictness=args.strictness,
        tools=tools or None,
        cache_size=0 if args.no_cache else 256,
    )

    # Read input
    items: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if args.max_items:
        items = items[:args.max_items]

    total = len(items)
    if total == 0:
        if not getattr(args, "quiet", False):
            print("No items to process.", file=sys.stderr)
        return

    if not getattr(args, "quiet", False):
        print(f"Processing {total} item(s)...", file=sys.stderr)

    results: list[dict] = []
    for idx, item in enumerate(items, 1):
        item_id = item.get("id", str(idx))
        prompt = item.get("prompt", "")
        if not prompt:
            if not getattr(args, "quiet", False):
                print(f"  [{idx}/{total}] Skipping item '{item_id}': no prompt", file=sys.stderr)
            continue

        if not getattr(args, "quiet", False):
            print(f"  [{idx}/{total}] Processing '{item_id}'...", file=sys.stderr)
        try:
            result = hallu.generate(model=args.model, prompt=prompt)
            result_dict = result.to_dict()
            result_dict["id"] = item_id
            result_dict["prompt"] = prompt
            results.append(result_dict)
        except Exception as exc:
            results.append({
                "id": item_id,
                "prompt": prompt,
                "error": f"{type(exc).__name__}: {exc}",
            })
            if not getattr(args, "quiet", False):
                print(f"  [{idx}/{total}] Error: {exc}", file=sys.stderr)

        if args.delay > 0 and idx < total:
            time.sleep(args.delay)

    # Write output
    output_lines = [json.dumps(r, ensure_ascii=False) for r in results]
    output = "\n".join(output_lines) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        if not getattr(args, "quiet", False):
            print(f"Batch results written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Summary to stderr
    errors = sum(1 for r in results if "error" in r)
    if not getattr(args, "quiet", False):
        print(f"Done: {len(results)} processed, {errors} error(s).", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__
        print(f"self-correct v{__version__}")
        return 0

    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "tools":
        return _cmd_tools()
    elif args.command == "models":
        return _cmd_models()
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "history":
        return _cmd_history()
    elif args.command == "info":
        cmd_info()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
