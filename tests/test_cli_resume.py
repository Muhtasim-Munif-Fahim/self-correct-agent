"""Tests for the ``resume`` CLI subcommand (saved verification sessions).

Batch ``--resume-from`` is a different feature; see ``test_batch_resume.py``.
"""

from __future__ import annotations

import json
from typing import Optional

from self_correct import cli, sessions
from self_correct.cli import _build_parser
from self_correct.core import AntiHallucinationResponse


def _write_session(tmp_path, *, name="session.json", prompt="Original prompt", config=None, result=None):
    path = tmp_path / name
    sessions.save_session(
        path,
        prompt=prompt,
        config=config
        or {
            "model": "saved-model",
            "strictness": 0.7,
            "provider": "ollama",
            "tools": ["wikipedia"],
            "max_retries": 2,
            "retry_backoff": 0.5,
            "max_calls": 9,
            "max_tokens": 400,
            "timeout": 30.0,
            "no_cache": False,
            "cache_ttl": 120.0,
            "cache_file": "claims.json",
            "checks": "checks.json",
            "base_url": "http://localhost:11434/v1",
            "api_key_env": "OLLAMA_KEY",
            "policy": ["policy.json"],
            "model_draft": "saved-draft",
            "model_extract": "saved-extract",
            "model_verify": "saved-verify",
            "model_correct": "saved-correct",
        },
        result=result or {"content": "Earlier result"},
    )
    return path


def _capture_verify(monkeypatch):
    captured = {}

    def fake_verify(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "cmd_verify", fake_verify)
    return captured


class _FakeHallucinator:
    """Offline stand-in used the same way other CLI tests mock the pipeline."""

    last_init: Optional[dict] = None
    last_generate: Optional[dict] = None

    def __init__(self, **kwargs):
        type(self).last_init = kwargs

    def generate(self, **kwargs):
        type(self).last_generate = kwargs
        return AntiHallucinationResponse(content="resumed output")


def _install_fake_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_CORRECT_HISTORY", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(cli, "_build_client", lambda args: object())
    _FakeHallucinator.last_init = None
    _FakeHallucinator.last_generate = None
    monkeypatch.setattr(cli, "AntiHallucinator", _FakeHallucinator)


def test_parser_accepts_resume_session_and_key_flags() -> None:
    args = _build_parser().parse_args(
        [
            "resume",
            "session.json",
            "--model",
            "gpt-4o",
            "--model-draft",
            "draft-model",
            "--strictness",
            "0.4",
            "--provider",
            "ollama",
            "--base-url",
            "http://localhost:11434/v1",
            "--api-key-env",
            "CUSTOM_KEY",
            "--max-retries",
            "3",
            "--retry-backoff",
            "1.5",
            "--max-calls",
            "5",
            "--checks",
            "c.json",
            "--output",
            "out.json",
            "--output-format",
            "json",
            "--include-log",
            "--save-session",
            "next.json",
            "--fail-on-hallucination",
        ]
    )
    assert args.command == "resume"
    assert args.session == "session.json"
    assert args.model == "gpt-4o"
    assert args.model_draft == "draft-model"
    assert args.strictness == 0.4
    assert args.provider == "ollama"
    assert args.base_url == "http://localhost:11434/v1"
    assert args.api_key_env == "CUSTOM_KEY"
    assert args.max_retries == 3
    assert args.retry_backoff == 1.5
    assert args.max_calls == 5
    assert args.checks == "c.json"
    assert args.output == "out.json"
    assert args.output_format == "json"
    assert args.include_log is True
    assert args.save_session == "next.json"
    assert args.fail_on_hallucination is True


def test_resume_loads_session_and_invokes_verify_with_saved_settings(
    tmp_path, monkeypatch
) -> None:
    path = _write_session(tmp_path)
    captured = _capture_verify(monkeypatch)

    assert cli.main(["resume", str(path)]) == 0

    assert captured["prompt"] == "Original prompt"
    assert captured["file"] is None
    assert captured["model"] == "saved-model"
    assert captured["strictness"] == 0.7
    assert captured["provider"] == "ollama"
    assert captured["tools"] == ["wikipedia"]
    assert captured["max_retries"] == 2
    assert captured["retry_backoff"] == 0.5
    assert captured["max_calls"] == 9
    assert captured["max_tokens"] == 400
    assert captured["timeout"] == 30.0
    assert captured["no_cache"] is False
    assert captured["cache_ttl"] == 120.0
    assert captured["cache_file"] == "claims.json"
    assert captured["checks"] == "checks.json"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key_env"] == "OLLAMA_KEY"
    assert captured["policy"] == ["policy.json"]
    assert captured["model_draft"] == "saved-draft"
    assert captured["model_extract"] == "saved-extract"
    assert captured["model_verify"] == "saved-verify"
    assert captured["model_correct"] == "saved-correct"
    assert captured["dry_run"] is False
    assert captured["fail_on_hallucination"] is False
    assert captured["save_session"] is None


def test_resume_overrides_win_over_saved_config(tmp_path, monkeypatch) -> None:
    path = _write_session(tmp_path)
    captured = _capture_verify(monkeypatch)

    assert (
        cli.main(
            [
                "resume",
                str(path),
                "--model",
                "new-model",
                "--strictness",
                "0.25",
                "--provider",
                "anthropic",
                "--model-draft",
                "override-draft",
                "--max-calls",
                "3",
                "--checks",
                "override-checks.json",
                "--save-session",
                "retry.json",
            ]
        )
        == 0
    )

    assert captured["prompt"] == "Original prompt"
    assert captured["model"] == "new-model"
    assert captured["strictness"] == 0.25
    assert captured["provider"] == "anthropic"
    assert captured["model_draft"] == "override-draft"
    assert captured["model_extract"] == "saved-extract"
    assert captured["max_calls"] == 3
    assert captured["checks"] == "override-checks.json"
    assert captured["tools"] == ["wikipedia"]
    assert captured["save_session"] == "retry.json"


def test_missing_session_file_exits_with_status_two(tmp_path, capsys) -> None:
    missing = tmp_path / "nope.json"
    assert cli.main(["resume", str(missing)]) == 2
    err = capsys.readouterr().err
    assert "resume:" in err
    assert str(missing) in err


def test_invalid_session_json_exits_with_status_two(tmp_path, capsys) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert cli.main(["resume", str(path)]) == 2
    err = capsys.readouterr().err
    assert "resume:" in err
    assert "not valid JSON" in err


def test_unsupported_session_schema_exits_with_status_two(tmp_path, capsys) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": 99, "prompt": "p"}), encoding="utf-8")
    assert cli.main(["resume", str(path)]) == 2
    err = capsys.readouterr().err
    assert "resume:" in err
    assert "unsupported session schema" in err


def test_resume_reruns_saved_prompt_through_mocked_pipeline(
    tmp_path, monkeypatch, capsys
) -> None:
    path = _write_session(tmp_path, prompt="Explain transformers.")
    _install_fake_pipeline(monkeypatch, tmp_path)
    report = tmp_path / "report.json"

    assert (
        cli.main(
            [
                "resume",
                str(path),
                "--model",
                "override-model",
                "--output",
                str(report),
                "--output-format",
                "json",
            ]
        )
        == 0
    )

    assert _FakeHallucinator.last_generate is not None
    assert _FakeHallucinator.last_generate["prompt"] == "Explain transformers."
    assert _FakeHallucinator.last_generate["model"] == "override-model"
    assert _FakeHallucinator.last_init["strictness"] == 0.7
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["content"] == "resumed output"
    assert "Report written to" in capsys.readouterr().out


def test_verify_save_session_then_resume_reruns_prompt(
    tmp_path, monkeypatch
) -> None:
    _install_fake_pipeline(monkeypatch, tmp_path)
    session = tmp_path / "saved.json"

    assert (
        cli.main(
            [
                "verify",
                "--prompt",
                "What is RLHF?",
                "--model",
                "saved-model",
                "--strictness",
                "0.6",
                "--save-session",
                str(session),
                "--output",
                str(tmp_path / "first.json"),
                "--output-format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(session.read_text(encoding="utf-8"))
    assert payload["prompt"] == "What is RLHF?"
    assert payload["config"]["model"] == "saved-model"
    assert payload["config"]["strictness"] == 0.6

    captured = _capture_verify(monkeypatch)
    assert cli.main(["resume", str(session), "--strictness", "0.9"]) == 0
    assert captured["prompt"] == "What is RLHF?"
    assert captured["model"] == "saved-model"
    assert captured["strictness"] == 0.9
