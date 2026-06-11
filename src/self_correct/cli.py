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
from typing import Optional

from .core import AntiHallucinator
from .tools import DuckDuckGoSearchTool, WikipediaSearchTool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self-correct",
        description="Anti-hallucination wrapper for LLMs using Chain-of-Verification.",
    )
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
        choices=["json", "markdown", "text"],
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

    # info subcommand
    info = sub.add_parser("info", help="Show package information")

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
                lines.append(f"  • {h}")
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


def cmd_info() -> None:
    """Show package information."""
    from . import __version__
    info = {
        "package": "self-correct",
        "version": __version__,
        "description": "A lightweight anti-hallucination wrapper for LLMs using Chain-of-Verification.",
    }
    print(json.dumps(info, indent=2))


def main(argv: Optional[list[str]] = None) -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "info":
        cmd_info()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
