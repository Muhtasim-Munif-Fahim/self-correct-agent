"""Command-line interface for self-correct-agent.

Provides batch verification workflows, file-based prompts,
and configurable output formats.

Usage::

    self-correct verify --model gpt-4o-mini --prompt "Explain quantum computing."
    self-correct verify --model gpt-4o-mini --file input.txt --output report.json
    self-correct verify --model gpt-4o-mini --file batch.txt --output-format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from . import history, sessions, templates
from .core import MODEL_PRICING, AntiHallucinator, VerificationPolicy, model_pricing
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
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
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
        "--file", "--prompt-file", "-f", default=None,
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
    verify.add_argument(
        "--cache-ttl", type=float, default=None, metavar="SECONDS",
        help="Expire cached verifications after SECONDS (default: never)",
    )
    verify.add_argument(
        "--cache-file",
        default=None,
        metavar="PATH",
        help="Load and update a persistent verified-claim cache",
    )
    verify.add_argument(
        "--max-tokens", type=int, default=None,
        help="Maximum tokens for the LLM response",
    )
    verify.add_argument(
        "--max-retries", type=int, default=0,
        help="Retry failed provider calls this many times (default: 0)",
    )
    verify.add_argument(
        "--retry-backoff", type=float, default=0.0, metavar="SECONDS",
        help="Initial delay between retries; delays double after each attempt",
    )
    verify.add_argument(
        "--timeout", type=float, default=None, metavar="SECONDS",
        help="Abort the run if the API does not respond within SECONDS",
    )
    verify.add_argument(
        "--provider", choices=["openai", "ollama", "custom"], default="openai",
        help="Where to send requests (default: openai)",
    )
    verify.add_argument(
        "--base-url", default=None,
        help="API base URL; required for --provider custom, overrides the default otherwise",
    )
    verify.add_argument(
        "--api-key-env", default=None, metavar="VAR",
        help="Environment variable holding the API key (default: OPENAI_API_KEY)",
    )
    verify.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without making any API calls",
    )
    verify.add_argument(
        "--save-session",
        default=None,
        metavar="PATH",
        help="Save the prompt, settings, and result so the run can be resumed",
    )
    verify.add_argument(
        "--fail-on-hallucination",
        action="store_true",
        help="Exit with status 1 when one or more claims are flagged",
    )
    verify.add_argument(
        "--policy",
        default=None,
        metavar="PATH",
        help="Apply a JSON verification policy and fail when it is not satisfied",
    )

    resume = sub.add_parser(
        "resume",
        help="Continue a verification from a saved session",
    )
    resume.add_argument("session", help="Session JSON created by --save-session")
    resume.add_argument("--model", default=None, help="Override the saved model")
    resume.add_argument(
        "--strictness",
        type=float,
        default=None,
        help="Override the saved verification strictness",
    )
    resume.add_argument(
        "--provider",
        choices=["openai", "ollama", "custom"],
        default=None,
        help="Override the saved provider",
    )
    resume.add_argument("--base-url", default=None, help="Override the saved API URL")
    resume.add_argument(
        "--api-key-env",
        default=None,
        metavar="VAR",
        help="Override the saved API-key environment variable",
    )
    resume.add_argument("--output", "-o", default=None, help="Write the new result to a file")
    resume.add_argument(
        "--output-format",
        choices=["json", "markdown", "text", "csv"],
        default=None,
    )
    resume.add_argument("--include-log", action="store_true")
    resume.add_argument(
        "--save-session",
        default=None,
        metavar="PATH",
        help="Save the resumed run as a new session",
    )
    resume.add_argument(
        "--fail-on-hallucination",
        action="store_true",
        help="Exit with status 1 when the resumed run flags a claim",
    )
    resume.add_argument(
        "--max-retries", type=int, default=None,
        help="Override the saved retry count",
    )
    resume.add_argument(
        "--retry-backoff", type=float, default=None, metavar="SECONDS",
        help="Override the saved retry backoff",
    )

    # tools subcommand
    tools_parser = sub.add_parser("tools", help="List available verification tools")

    # models subcommand
    models_parser = sub.add_parser("models", help="List supported models with estimated costs")

    # history subcommand
    history_parser = sub.add_parser("history", help="Show recent verification runs")
    history_parser.add_argument(
        "--limit", "-n", type=int, default=20,
        help="Number of runs to show, most recent first (default: 20)",
    )
    history_parser.add_argument(
        "--export", default=None, metavar="PATH",
        help="Write the full history to PATH (.json or .jsonl)",
    )
    history_parser.add_argument(
        "--clear", action="store_true",
        help="Delete the recorded history",
    )

    stats_parser = sub.add_parser("stats", help="Aggregate statistics across recorded runs")
    stats_parser.add_argument(
        "--json", action="store_true", help="Print the summary as JSON",
    )

    cache_parser = sub.add_parser("cache", help="Show claim-cache configuration and effectiveness")
    cache_parser.add_argument(
        "--json", action="store_true", help="Print the summary as JSON",
    )

    compare_parser = sub.add_parser(
        "compare", aliases=["diff"],
        help="Compare two verification results saved with --output-format json",
    )
    compare_parser.add_argument("baseline", help="Earlier result JSON")
    compare_parser.add_argument("current", help="Later result JSON")
    compare_parser.add_argument(
        "--model", default=None,
        help="Model to price token differences with (default: read from the results)",
    )

    template_parser = sub.add_parser(
        "template", aliases=["prompt"],
        help="List, show and render built-in and user prompt templates",
    )
    template_sub = template_parser.add_subparsers(dest="template_command", required=True)
    template_sub.add_parser("list", help="List available templates")
    template_show = template_sub.add_parser("show", help="Print a template body")
    template_show.add_argument("name", help="Template name")
    template_render = template_sub.add_parser("render", help="Fill a template's placeholders")
    template_render.add_argument("name", help="Template name")
    template_render.add_argument(
        "--var", action="append", default=[], metavar="KEY=VALUE",
        help="Value for a placeholder; repeatable",
    )

    # estimate subcommand
    estimate = sub.add_parser("estimate", help="Estimate token count for a prompt")
    estimate.add_argument(
        "--prompt", "-p", required=True,
        help="Prompt text to estimate tokens for",
    )

    # config subcommand
    config = sub.add_parser("config", help="Manage configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_init = config_sub.add_parser("init", help="Generate a config template")
    config_init.add_argument(
        "--output", "-o", default="self-correct.json",
        help="Output path for the config file (default: self-correct.json)",
    )
    config_validate = config_sub.add_parser("validate", help="Validate a config file")
    config_validate.add_argument(
        "--config", "-c", default="self-correct.json",
        help="Path to the config file (default: self-correct.json)",
    )

    # upgrade subcommand
    upgrade = sub.add_parser("upgrade", help="Suggest upgrading to the latest version")

    # info subcommand
    info = sub.add_parser("info", help="Show package information")

    # batch subcommand
    batch = sub.add_parser("batch", help="Process multiple prompts from a JSONL file")
    batch.add_argument(
        "--input", "-i", default=None,
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
        "--cache-ttl", type=float, default=None, metavar="SECONDS",
        help="Expire cached verifications after SECONDS (default: never)",
    )
    batch.add_argument(
        "--timeout", type=float, default=None, metavar="SECONDS",
        help="Abort the run if the API does not respond within SECONDS",
    )
    batch.add_argument(
        "--delay", type=float, default=0.0,
        help="Delay in seconds between items (to avoid rate limits)",
    )
    batch.add_argument(
        "--format", choices=["json", "jsonl"], default="jsonl",
        help="Output format (default: jsonl)",
    )
    batch.add_argument(
        "--provider", choices=["openai", "ollama", "custom"], default="openai",
        help="Where to send requests (default: openai)",
    )
    batch.add_argument(
        "--base-url", default=None,
        help="API base URL; required for --provider custom, overrides the default otherwise",
    )
    batch.add_argument(
        "--api-key-env", default=None, metavar="VAR",
        help="Environment variable holding the API key (default: OPENAI_API_KEY)",
    )
    batch.add_argument(
        "--schema", action="store_true",
        help="Print the output record schema and exit without processing",
    )
    batch.add_argument(
        "--fail-on-hallucination",
        action="store_true",
        help="Exit with status 1 when any processed item flags a claim",
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


#: Default endpoint for each provider. `custom` has none by design: it exists
#: precisely for endpoints this package does not know about.
PROVIDER_BASE_URLS = {
    "openai": None,
    "ollama": "http://localhost:11434/v1",
    "custom": None,
}


def _build_client(args: argparse.Namespace):
    """Construct the API client for the selected provider.

    Every supported provider speaks the OpenAI chat-completions protocol, so
    one client class covers all of them and only the base URL and key differ.
    That is also what the core requires: it calls
    `client.chat.completions.create()` and nothing else.
    """

    import os

    from openai import OpenAI

    provider = getattr(args, "provider", "openai") or "openai"
    base_url = getattr(args, "base_url", None) or PROVIDER_BASE_URLS.get(provider)

    if provider == "custom" and not base_url:
        raise SystemExit("--provider custom requires --base-url")

    key_env = getattr(args, "api_key_env", None) or "OPENAI_API_KEY"
    api_key = os.environ.get(key_env)
    if not api_key:
        # A local Ollama server ignores the key but the client insists on one.
        if provider == "ollama":
            api_key = "ollama"
        else:
            raise SystemExit(
                f"No API key found in ${key_env}. Set it, or pass --api-key-env "
                "to name a different variable."
            )

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    timeout = getattr(args, "timeout", None)
    if timeout:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def _format_cost(usage: object, model: str) -> str:
    """Render an estimated USD cost, or say plainly that it is unknown."""

    estimate = getattr(usage, "estimate_cost_for_model", None)
    cost = estimate(model) if callable(estimate) else None
    if cost is None:
        return f"unknown (no published rate for '{model}')"
    # Sub-cent runs are the norm here, so a two-decimal figure would read $0.00
    # for every run and tell the user nothing. Below the smallest figure four
    # decimals can show, say so rather than printing $0.0000, which reads as
    # free.
    if cost > 0 and cost < 0.0001:
        return "<$0.0001"
    return f"${cost:.4f}" if cost < 0.01 else f"${cost:.2f}"


def _print_dry_run_plan(
    args: argparse.Namespace, prompt: str, tool_names: list) -> None:
    """Describe the run that --dry-run is standing in for.

    Nothing here touches the network, so it works without an API key and
    without spending anything — the point is to confirm the prompt, model and
    tools are what you meant before paying for a real run.
    """

    preview = prompt if len(prompt) <= 200 else prompt[:200] + "..."
    est_tokens = max(1, len(prompt) // 4)

    print("DRY RUN - no API calls will be made")
    print("-" * 58)
    print(f"{'Model':<18}{args.model}")
    print(f"{'Strictness':<18}{args.strictness}")
    print(f"{'Tools':<18}{', '.join(tool_names) if tool_names else 'none'}")
    print(f"{'Cache':<18}{'disabled' if args.no_cache else 'enabled'}"
          + (f" (ttl {args.cache_ttl}s)" if getattr(args, "cache_ttl", None) else ""))
    print(f"{'Max tokens':<18}{args.max_tokens if args.max_tokens is not None else 'model default'}")
    print(f"{'Timeout':<18}{f'{args.timeout}s' if getattr(args, 'timeout', None) else 'none'}")
    print(f"{'Output':<18}{args.output or 'stdout'}"
          f" ({_detect_output_format(args.output, args.output_format)})")
    print()
    print(f"Prompt ({len(prompt)} chars, ~{est_tokens} tokens):")
    print(f"  {preview}")
    print()
    print("Would then: generate a draft, extract claims, verify each one"
          + (" using the tools above" if tool_names else " against the model")
          + ", and revise the draft.")


def _record_verify_run(
    args: argparse.Namespace,
    prompt: str,
    result: object,
    duration: float,
    hallu: object,
) -> None:
    """Persist one verify run so history, stats and cache can report on it.

    Every field is read defensively: recording is a side benefit and must never
    be the reason a completed verification fails.
    """

    entry = {
        "command": "verify",
        "model": args.model,
        "prompt": prompt[:120],
        "duration": duration,
        "strictness": args.strictness,
        "tools": list(args.tools or []),
    }

    usage = getattr(result, "token_usage", None)
    if usage is not None:
        entry["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
        entry["completion_tokens"] = getattr(usage, "completion_tokens", 0)

    log = getattr(result, "verification_log", None)
    if isinstance(log, list):
        entry["claims"] = len(log)
        entry["claims_verified"] = sum(1 for item in log if item.get("valid"))

    cache = getattr(hallu, "_cache", None)
    stats = getattr(cache, "stats", None)
    if callable(stats):
        cache_stats = stats()
        entry["cache_hits"] = cache_stats.get("hits", 0)
        entry["cache_misses"] = cache_stats.get("misses", 0)

    history.record_run(entry)


def cmd_verify(args: argparse.Namespace) -> None:
    """Execute the verify subcommand."""
    from openai import OpenAI

    prompt = _read_prompt(args.prompt, args.file)

    tool_names = [name for name in ("duckduckgo", "wikipedia") if name in args.tools]

    if getattr(args, "dry_run", False):
        _print_dry_run_plan(args, prompt, tool_names)
        return

    # The timeout belongs on the client so it covers every request the
    # verification pipeline makes, not just the first generation call.
    client = _build_client(args)

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
        cache_ttl=getattr(args, "cache_ttl", None),
        max_retries=getattr(args, "max_retries", 0),
        retry_backoff=getattr(args, "retry_backoff", 0.0),
    )
    cache_file = getattr(args, "cache_file", None)
    if cache_file and args.no_cache:
        raise ValueError("--cache-file cannot be combined with --no-cache")
    if cache_file and Path(cache_file).exists():
        hallu.load_cache(cache_file)

    started = time.time()
    try:
        result = hallu.generate(model=args.model, prompt=prompt)
    except Exception as exc:
        history.record_run({
            "command": "verify",
            "model": args.model,
            "prompt": prompt[:120],
            "duration": time.time() - started,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    _record_verify_run(args, prompt, result, time.time() - started, hallu)
    if cache_file:
        hallu.save_cache(cache_file)

    session_path = getattr(args, "save_session", None)
    if session_path:
        sessions.save_session(
            session_path,
            prompt=prompt,
            config={
                "model": args.model,
                "strictness": args.strictness,
                "tools": list(args.tools or []),
                "no_cache": bool(args.no_cache),
                "cache_ttl": getattr(args, "cache_ttl", None),
                "cache_file": cache_file,
                "max_tokens": getattr(args, "max_tokens", None),
                "max_retries": getattr(args, "max_retries", 0),
                "retry_backoff": getattr(args, "retry_backoff", 0.0),
                "timeout": getattr(args, "timeout", None),
                "provider": getattr(args, "provider", "openai"),
                "base_url": getattr(args, "base_url", None),
                "api_key_env": getattr(args, "api_key_env", None),
                "policy": getattr(args, "policy", None),
            },
            result=result.to_dict(),
        )

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
            f"Tokens: {result.token_usage.total_tokens}"
            f" (prompt {result.token_usage.prompt_tokens},"
            f" completion {result.token_usage.completion_tokens})",
            f"Estimated cost: {_format_cost(result.token_usage, args.model)}",
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
    policy = (
        VerificationPolicy.from_json(args.policy)
        if getattr(args, "policy", None)
        else None
    )
    decision = result.evaluate(policy) if policy is not None else None
    if decision is not None and not decision.passed:
        print("Verification policy failed: " + "; ".join(decision.reasons), file=sys.stderr)
    return _verification_exit_code(
        result,
        fail_on_hallucination=getattr(args, "fail_on_hallucination", False),
        policy=policy,
    )


def _verification_exit_code(
    result: object,
    *,
    fail_on_hallucination: bool,
    policy: VerificationPolicy | None = None,
) -> int:
    """Translate verification findings into an opt-in CI exit code."""

    flagged = getattr(result, "hallucinations_caught", None) or []
    if fail_on_hallucination and flagged:
        return 1
    if policy is not None and not policy.evaluate(result).passed:
        return 1
    return 0


def _cmd_resume(args: argparse.Namespace) -> int | None:
    """Re-run a saved prompt with its original settings and optional overrides."""

    try:
        session = sessions.load_session(args.session)
    except ValueError as exc:
        print(f"resume: {exc}", file=sys.stderr)
        return 2

    config = session["config"]
    verify_args = argparse.Namespace(
        prompt=session["prompt"],
        file=None,
        model=args.model or config.get("model", "gpt-4o-mini"),
        strictness=(
            args.strictness
            if args.strictness is not None
            else config.get("strictness", 1.0)
        ),
        tools=list(config.get("tools") or []),
        output=args.output,
        output_format=args.output_format,
        include_log=args.include_log,
        no_cache=bool(config.get("no_cache", False)),
        cache_ttl=config.get("cache_ttl"),
        max_tokens=config.get("max_tokens"),
        max_retries=(
            args.max_retries
            if args.max_retries is not None
            else config.get("max_retries", 0)
        ),
        retry_backoff=(
            args.retry_backoff
            if args.retry_backoff is not None
            else config.get("retry_backoff", 0.0)
        ),
        timeout=config.get("timeout"),
        provider=args.provider or config.get("provider", "openai"),
        base_url=(args.base_url if args.base_url is not None else config.get("base_url")),
        api_key_env=(
            args.api_key_env
            if args.api_key_env is not None
            else config.get("api_key_env")
        ),
        policy=config.get("policy"),
        dry_run=False,
        save_session=args.save_session,
        fail_on_hallucination=args.fail_on_hallucination,
        verbose=getattr(args, "verbose", False),
        quiet=getattr(args, "quiet", False),
    )
    return cmd_verify(verify_args)


def cmd_estimate(args: argparse.Namespace) -> None:
    """Estimate token count for a prompt."""
    prompt = args.prompt
    char_count = len(prompt)
    word_count = len(prompt.split())
    estimated_tokens = int(char_count / 4)
    print(f"Characters: {char_count}")
    print(f"Words: {word_count}")
    print(f"Estimated tokens: ~{estimated_tokens}")


def cmd_config_init(args: argparse.Namespace) -> None:
    """Generate a config template."""
    template = {
        "model": "gpt-4o-mini",
        "strictness": 1.0,
        "tools": [],
        "no_cache": False,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
        f.write("\n")
    print(f"Config template written to {args.output}")


def cmd_upgrade() -> None:
    """Suggest upgrading to the latest version."""
    print("Run: pip install --upgrade self-correct-agent")


#: Tokens a representative verification run consumes, used to turn per-token
#: rates into a figure that means something. A Chain-of-Verification run makes
#: several calls, so completion volume is closer to prompt volume than a single
#: chat completion would suggest.
_TYPICAL_RUN_TOKENS = (2000, 800)


def cmd_info() -> None:
    """Show package information and per-model cost estimates."""
    from . import __version__

    info = {
        "package": "self-correct",
        "version": __version__,
        "description": "A lightweight anti-hallucination wrapper for LLMs using Chain-of-Verification.",
    }
    print(json.dumps(info, indent=2))

    prompt_tokens, completion_tokens = _TYPICAL_RUN_TOKENS
    print()
    print(
        f"Estimated cost per verification run "
        f"(~{prompt_tokens} prompt + {completion_tokens} completion tokens):"
    )
    print(f"  {'Model':<20}{'Per run':>12}{'Per 100 runs':>16}")
    print("  " + "-" * 46)
    for model, (prompt_rate, completion_rate) in MODEL_PRICING.items():
        per_run = (
            (prompt_tokens / 1_000_000) * prompt_rate
            + (completion_tokens / 1_000_000) * completion_rate
        )
        print(f"  {model:<20}{'$' + format(per_run, '.4f'):>12}{'$' + format(per_run * 100, '.2f'):>16}")
    print()
    print("  Rates are approximate and set by the provider; a run's real cost")
    print("  depends on prompt length and how many claims need verifying.")


def _cmd_history(args: argparse.Namespace) -> int:
    """Show, export or clear the recorded run history."""
    path = history.history_path()

    if args.clear:
        try:
            path.unlink()
            print(f"Cleared history at {path}")
        except FileNotFoundError:
            print("No history to clear.")
        except OSError as exc:
            print(f"Could not clear history: {exc}", file=sys.stderr)
            return 2
        return 0

    runs = history.load_runs()
    if not runs:
        print(f"No runs recorded yet. History is written to {path}")
        return 0

    if args.export:
        try:
            with open(args.export, "w", encoding="utf-8") as handle:
                if args.export.endswith(".json"):
                    json.dump(runs, handle, indent=2, default=str)
                else:
                    for run in runs:
                        handle.write(json.dumps(run, default=str) + "\n")
        except OSError as exc:
            print(f"Could not write '{args.export}': {exc}", file=sys.stderr)
            return 2
        print(f"Exported {len(runs)} run(s) to {args.export}")
        return 0

    recent = runs[-args.limit:][::-1]
    print(f"{'When':<20}{'Model':<16}{'Claims':>8}{'Verified':>10}{'Secs':>8}  Prompt")
    print("-" * 92)
    for run in recent:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(run.get("timestamp", 0))))
        claims = run.get("claims", "-")
        verified = run.get("claims_verified", "-")
        duration = run.get("duration")
        secs = f"{float(duration):.1f}" if duration is not None else "-"
        preview = str(run.get("prompt", ""))[:34]
        marker = "  [error]" if run.get("error") else ""
        print(f"{when:<20}{str(run.get('model','?')):<16}{claims:>8}{verified:>10}{secs:>8}  {preview}{marker}")
    print()
    print(f"{len(runs)} run(s) recorded at {path}")
    return 0


def _cmd_template(args: argparse.Namespace) -> int:
    """List, show or render a prompt template."""
    if args.template_command == "list":
        rows = templates.list_templates()
        print(f"{'Name':<24}{'Source':<26}Description")
        print("-" * 92)
        for name, source, description in rows:
            print(f"{name:<24}{source:<26}{description}")
        print()
        print(f"User templates are read from {templates.template_dir()}")
        print("Add a .txt file there to define one; a matching name overrides a built-in.")
        return 0

    body = templates.get_template(args.name)
    if body is None:
        available = ", ".join(name for name, _, _ in templates.list_templates())
        print(f"No template named '{args.name}'. Available: {available}", file=sys.stderr)
        return 2

    if args.template_command == "show":
        expected = templates.placeholders(body)
        print(body)
        if expected:
            print()
            print(f"Placeholders: {', '.join('$' + name for name in expected)}")
        return 0

    values = {}
    for pair in args.var:
        if "=" not in pair:
            print(f"Invalid --var '{pair}'; expected KEY=VALUE.", file=sys.stderr)
            return 2
        key, value = pair.split("=", 1)
        values[key.strip()] = value

    rendered, missing = templates.render(body, values)
    if missing:
        print(
            "Missing value(s) for: " + ", ".join("$" + name for name in missing)
            + ". Pass them with --var NAME=VALUE.",
            file=sys.stderr,
        )
        return 2
    print(rendered)
    return 0


def _load_result(path: str, label: str):
    """Read one verification result written with --output-format json."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except OSError as exc:
        return None, f"Cannot read {label} result '{path}': {exc}"
    except json.JSONDecodeError as exc:
        return None, (
            f"{label.capitalize()} result '{path}' is not valid JSON: {exc}. "
            "Results come from `verify --output-format json`."
        )


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare two verification results side by side."""
    before, error = _load_result(args.baseline, "baseline")
    if error:
        print(error, file=sys.stderr)
        return 2
    after, error = _load_result(args.current, "current")
    if error:
        print(error, file=sys.stderr)
        return 2

    def _tokens(result, key):
        usage = result.get("token_usage") or {}
        return int(usage.get(key, 0) or 0)

    model = args.model or after.get("model") or before.get("model") or "gpt-4o-mini"

    print(f"{'Field':<24}{'baseline':>14}{'current':>14}{'change':>12}")
    print("-" * 64)
    print(f"{'status':<24}{str(before.get('status','?')):>14}{str(after.get('status','?')):>14}{'':>12}")

    for label, key in (("prompt tokens", "prompt_tokens"),
                       ("completion tokens", "completion_tokens"),
                       ("total tokens", "total_tokens")):
        old, new = _tokens(before, key), _tokens(after, key)
        print(f"{label:<24}{old:>14}{new:>14}{new - old:>+12}")

    rates = model_pricing(model)
    if rates:
        prompt_rate, completion_rate = rates
        def _cost(result):
            return ((_tokens(result, "prompt_tokens") / 1_000_000) * prompt_rate
                    + (_tokens(result, "completion_tokens") / 1_000_000) * completion_rate)
        old_cost, new_cost = _cost(before), _cost(after)
        print(f"{'est. cost (' + model + ')':<24}"
              f"{'$' + format(old_cost, '.4f'):>14}{'$' + format(new_cost, '.4f'):>14}"
              f"{'$' + format(new_cost - old_cost, '+.4f'):>12}")

    old_flags = {str(x) for x in (before.get("hallucinations_caught") or [])}
    new_flags = {str(x) for x in (after.get("hallucinations_caught") or [])}
    print(f"{'flagged claims':<24}{len(old_flags):>14}{len(new_flags):>14}{len(new_flags) - len(old_flags):>+12}")

    added, resolved = sorted(new_flags - old_flags), sorted(old_flags - new_flags)
    if added or resolved:
        print()
        print("Flagged-claim changes:")
        for claim in added:
            print(f"  + {claim[:88]}")
        for claim in resolved:
            print(f"  - {claim[:88]}")

    old_text = str(before.get("content", ""))
    new_text = str(after.get("content", ""))
    print()
    if old_text == new_text:
        print("Final output: identical")
    else:
        print(f"Final output: differs ({len(old_text)} -> {len(new_text)} chars)")

    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Aggregate statistics across every recorded run."""
    summary = history.aggregate(history.load_runs())

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if not summary.get("runs"):
        print("No runs recorded yet.")
        return 0

    def _row(label: str, value: object) -> None:
        print(f"{label:<26}{value}")

    _row("Runs", summary["runs"])
    _row("Runs with errors", summary["errors"])
    _row("First run", time.strftime("%Y-%m-%d %H:%M", time.localtime(summary["first"])))
    _row("Last run", time.strftime("%Y-%m-%d %H:%M", time.localtime(summary["last"])))
    print()
    _row("Claims extracted", summary["claims"])
    _row("Claims verified", f"{summary['claims_verified']} ({summary['verified_rate']:.1%})")
    print()
    _row("Prompt tokens", f"{summary['prompt_tokens']:,}")
    _row("Completion tokens", f"{summary['completion_tokens']:,}")
    _row("Total time", f"{summary['total_duration']:.1f}s")
    _row("Mean per run", f"{summary['mean_duration']:.1f}s")
    print()
    print("Models used")
    for model, count in summary["models"].items():
        print(f"  {model:<24}{count}")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    """Report claim-cache effectiveness across recorded runs."""
    summary = history.aggregate(history.load_runs())
    hits = summary.get("cache_hits", 0)
    misses = summary.get("cache_misses", 0)
    looked_up = hits + misses
    payload = {
        "runs": summary.get("runs", 0),
        "hits": hits,
        "misses": misses,
        "hit_rate": (hits / looked_up) if looked_up else 0.0,
        "lookups": looked_up,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("The claim cache lives in memory for the duration of a run, so these")
    print("figures are accumulated from recorded runs rather than read live.")
    print("-" * 62)
    print(f"{'Runs recorded':<22}{payload['runs']}")
    print(f"{'Claim lookups':<22}{payload['lookups']}")
    print(f"{'Hits':<22}{payload['hits']}")
    print(f"{'Misses':<22}{payload['misses']}")
    print(f"{'Hit rate':<22}{payload['hit_rate']:.1%}")
    if not payload["lookups"]:
        print()
        print("No lookups recorded yet - run `self-correct verify` first.")
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
    print(f"{'Model':<25} {'Input/1M':<15} {'Output/1M':<15}")
    print("-" * 55)
    for name, (prompt_rate, completion_rate) in MODEL_PRICING.items():
        print(f"{name:<25} {'$' + format(prompt_rate, '.2f'):<15} {'$' + format(completion_rate, '.2f'):<15}")
    print()
    print("* Prices per 1M tokens, approximate. Check provider for current pricing.")
    return 0


#: Fields written for each batch record, as (name, type, description).
_BATCH_OUTPUT_SCHEMA = [
    ("id", "string", "Item id from the input, or its 1-based position"),
    ("prompt", "string", "The prompt as submitted"),
    ("content", "string", "Final verified text (absent when the item errored)"),
    ("status", "string", "Verification outcome for the item"),
    ("hallucinations_caught", "array[string]", "Claims the pipeline rejected"),
    ("verification_log", "array[object]", "Per-claim record: claim, valid, source"),
    ("token_usage", "object", "prompt_tokens, completion_tokens, total_tokens"),
    ("elapsed_seconds", "number", "Wall-clock time for the item"),
    ("error", "string", "Present only when the item failed; type and message"),
]


def _print_batch_schema(args: argparse.Namespace) -> None:
    """Describe the records `batch` will write."""

    print(f"Output format: {args.format}")
    if args.format == "jsonl":
        print("One JSON object per line, one line per input item.")
    else:
        print("A single JSON array of objects, one per input item.")
    print()
    print(f"{'Field':<24}{'Type':<18}Description")
    print("-" * 92)
    for name, type_name, description in _BATCH_OUTPUT_SCHEMA:
        print(f"{name:<24}{type_name:<18}{description}")
    print()
    print("Every item yields a record. A failed item carries `error` in place of")
    print("`content`, so the output line count always matches the input.")


def cmd_batch(args: argparse.Namespace) -> None:
    """Execute the batch subcommand: process a JSONL file."""
    if getattr(args, "schema", False):
        _print_batch_schema(args)
        return

    # --input is only optional so that --schema can run without one.
    if not args.input:
        raise SystemExit("batch: --input is required (or use --schema to see the output fields)")

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
        cache_ttl=getattr(args, "cache_ttl", None),
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
            generate_kwargs = {"model": args.model, "prompt": prompt}
            if args.max_tokens is not None:
                generate_kwargs["max_tokens"] = args.max_tokens
            result = hallu.generate(**generate_kwargs)
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
    if getattr(args, "format", "jsonl") == "json":
        output = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    else:
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
    flagged = sum(bool(r.get("hallucinations_caught")) for r in results)
    return 1 if getattr(args, "fail_on_hallucination", False) and flagged else 0


def cmd_config_validate(args: argparse.Namespace) -> None:
    """Validate a config file."""
    import os
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        required_keys = {"model"}
        missing = required_keys - set(config.keys())
        if missing:
            print(f"Validation failed: missing required keys: {missing}")
            return
        print(f"Config file '{config_path}' is valid.")
    except json.JSONDecodeError as e:
        print(f"Validation failed: invalid JSON - {e}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__
        print(f"self-correct v{__version__}")
        return 0

    if args.command == "verify":
        return cmd_verify(args)
    elif args.command == "resume":
        return _cmd_resume(args)
    elif args.command == "tools":
        return _cmd_tools()
    elif args.command == "models":
        return _cmd_models()
    elif args.command == "batch":
        return cmd_batch(args)
    elif args.command == "history":
        return _cmd_history(args)
    elif args.command in ("template", "prompt"):
        return _cmd_template(args)
    elif args.command in ("compare", "diff"):
        return _cmd_compare(args)
    elif args.command == "stats":
        return _cmd_stats(args)
    elif args.command == "cache":
        return _cmd_cache(args)
    elif args.command == "info":
        cmd_info()
    elif args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "config":
        if args.config_command == "init":
            cmd_config_init(args)
        elif args.config_command == "validate":
            cmd_config_validate(args)
    elif args.command == "upgrade":
        cmd_upgrade()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
