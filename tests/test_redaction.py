"""Tests for masking sensitive spans before reports are persisted."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from self_correct import cli
from self_correct.redaction import DEFAULT_REPLACEMENT, SecretRedactor, load_redaction_rules


def _redactor(*patterns, replacement=DEFAULT_REPLACEMENT):
    return SecretRedactor([(re.compile(p), replacement) for p in patterns])


def test_matched_spans_become_the_placeholder() -> None:
    redactor = _redactor(r"sk-[A-Za-z0-9]{6,}")
    assert redactor.redact("using key sk-abc123def now") == "using key [REDACTED] now"


def test_every_rule_is_applied_in_order() -> None:
    redactor = _redactor(r"sk-\w+", r"user@\w+\.com")
    masked = redactor.redact("sk-secret1 from user@corp.com")
    assert "sk-secret1" not in masked
    assert "user@corp.com" not in masked


def test_case_insensitive_flag_controls_matching() -> None:
    plain = _redactor(r"token=\w+")
    insensitive = SecretRedactor(
        [(re.compile(r"token=\w+", re.IGNORECASE), "[REDACTED]")]
    )
    assert "TOKEN=abc" in plain.redact("TOKEN=abc")
    assert "TOKEN=abc" not in insensitive.redact("TOKEN=abc")


def test_custom_default_replacement_is_used() -> None:
    redactor = SecretRedactor(
        [(re.compile(r"\d{4}"), "@@@")], default_replacement="@@@"
    )
    assert redactor.redact("pin 1234 ok") == "pin @@@ ok"


def test_unmatched_text_survives_verbatim() -> None:
    redactor = _redactor(r"nope-never")
    text = "ordinary report text"
    assert redactor.redact(text) == text


def test_empty_span_patterns_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not match an empty span"):
        SecretRedactor([(re.compile(r"x*"), "[REDACTED]")])


def test_replacements_must_be_non_empty_strings() -> None:
    with pytest.raises(ValueError, match="replacement"):
        SecretRedactor([(re.compile(r"x"), "")])
    with pytest.raises(ValueError, match="default replacement"):
        SecretRedactor([], default_replacement="")


def test_loader_reads_rules_from_json(tmp_path) -> None:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({
        "rules": [
            {"name": "api-key", "pattern": r"sk-[a-z0-9]+"},
            {
                "name": "internal-host",
                "pattern": r"db-\d+\.internal",
                "replacement": "<host>",
                "flags": ["IGNORECASE"],
            },
        ],
    }), encoding="utf-8")

    redactor = load_redaction_rules(path)
    masked = redactor.redact("leak sk-abc123 and DB-42.INTERNAL here")

    assert masked == f"leak {DEFAULT_REPLACEMENT} and <host> here"


def test_loader_top_level_replacement_covers_rules_without_one(tmp_path) -> None:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({
        "replacement": "***",
        "rules": [{"name": "digits", "pattern": r"\d+"}],
    }), encoding="utf-8")

    assert load_redaction_rules(path).redact("room 12") == "room ***"


def test_loader_validates_structure(tmp_path) -> None:
    bad_root = tmp_path / "root.json"
    bad_root.write_text(json.dumps({"checks": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="'rules' list"):
        load_redaction_rules(bad_root)

    nameless = tmp_path / "nameless.json"
    nameless.write_text(json.dumps({"rules": [{"pattern": "x"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="name"):
        load_redaction_rules(nameless)

    unknown_flag = tmp_path / "flag.json"
    unknown_flag.write_text(
        json.dumps({"rules": [{"name": "n", "pattern": "x", "flags": ["BOGUS"]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown regex flag"):
        load_redaction_rules(unknown_flag)


class _Usage:
    total_tokens = 10
    prompt_tokens = 4
    completion_tokens = 6


class _Response:
    def __init__(self, content):
        self.content = content
        self.hallucinations_caught = []
        self.verification_log = []
        self.token_usage = _Usage()
        self.elapsed_seconds = 0.1

    def to_dict(self):
        return {"content": self.content}

    def to_markdown(self, include_log=False):
        return f"# Report\n\n{self.content}"

    def evaluate(self, policy):
        return SimpleNamespace(passed=True, reasons=[])


class _StubAntiHallucinator:
    def __init__(self, **kwargs):
        self.response = _Response(kwargs.get("_content", ""))

    def generate(self, **kwargs):
        return self.response


@pytest.fixture()
def offline_verify(monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_CORRECT_HISTORY", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(cli, "_build_client", lambda args: object())
    monkeypatch.setattr(cli, "AntiHallucinator", _StubAntiHallucinator)
    return tmp_path


def test_verify_masks_secrets_before_persisting_the_report(
    offline_verify, tmp_path, capsys
) -> None:
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps({"rules": [{"name": "api-key", "pattern": r"sk-\w+"}]}),
        encoding="utf-8",
    )
    report = tmp_path / "report.md"

    _StubAntiHallucinator._content_override = "The key was sk-topsecret123."
    original_init = _StubAntiHallucinator.__init__

    def init_with_secret(self, **kwargs):
        kwargs["_content"] = "The key was sk-topsecret123."
        original_init(self, **kwargs)

    _StubAntiHallucinator.__init__ = init_with_secret
    try:
        exit_code = cli.main([
            "verify", "--prompt", "Summarise using my key",
            "--redact", str(rules), "--output", str(report),
        ])
    finally:
        _StubAntiHallucinator.__init__ = original_init

    written = report.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "sk-topsecret123" not in written
    assert "[REDACTED]" in written


def test_verify_without_redaction_keeps_report_as_generated(
    offline_verify, tmp_path
) -> None:
    report = tmp_path / "report.md"
    original_init = _StubAntiHallucinator.__init__

    def init_with_secret(self, **kwargs):
        kwargs["_content"] = "The key was sk-visible123."
        original_init(self, **kwargs)

    _StubAntiHallucinator.__init__ = init_with_secret
    try:
        cli.main(["verify", "--prompt", "p", "--output", str(report)])
    finally:
        _StubAntiHallucinator.__init__ = original_init

    assert "sk-visible123" in report.read_text(encoding="utf-8")


def test_verify_rejects_an_invalid_redaction_file(offline_verify, capsys) -> None:
    broken = offline_verify / "broken.json"
    broken.write_text("{", encoding="utf-8")

    with pytest.raises(SystemExit, match="--redact"):
        cli.main(["verify", "--prompt", "p", "--redact", str(broken)])