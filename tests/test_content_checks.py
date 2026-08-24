"""Tests for pluggable content checks."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from self_correct.cli import _build_parser
from self_correct.core import (
    AntiHallucinator,
    ContentCheck,
    RegexContentCheck,
    load_content_checks,
)


def _mock_response(content, prompt_tokens=5, completion_tokens=5):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


class UppercaseCheck(ContentCheck):
    @property
    def name(self):
        return "no-shouting"

    def check(self, content):
        words = [w for w in content.split() if w.isupper() and len(w) > 3]
        return [f"shouting word: {w}" for w in words]


def test_content_check_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ContentCheck()  # type: ignore[abstract]


def test_regex_check_reports_each_match() -> None:
    check = RegexContentCheck("banned", r"definitely|absolutely")
    assert check.name == "banned"
    assert check.check("definitely true and absolutely false") == [
        "definitely",
        "absolutely",
    ]
    assert check.check("nothing here") == []


def test_regex_check_flags_are_honored() -> None:
    import re

    check = RegexContentCheck("caseless", r"secret", flags=re.IGNORECASE)
    assert check.check("My SECRET plan") == ["SECRET"]


def test_regex_check_rejects_blank_names_and_bad_patterns() -> None:
    with pytest.raises(ValueError, match="name"):
        RegexContentCheck("  ", r"x")
    import re

    with pytest.raises(re.error):
        RegexContentCheck("broken", r"([unclosed")


def test_regex_check_caps_reported_findings() -> None:
    check = RegexContentCheck("spam", r"a")
    findings = check.check("a" * 50)
    assert len(findings) == RegexContentCheck.MAX_FINDINGS


def test_agent_rejects_non_check_entries() -> None:
    with pytest.raises(ValueError, match="ContentCheck instances"):
        AntiHallucinator(MagicMock(), content_checks=["not-a-check"])


def _scripted_client(*contents):
    client = MagicMock()
    client.chat.completions.create.side_effect = [_mock_response(c) for c in contents]
    return client


def test_violations_become_flagged_verdicts() -> None:
    client = _scripted_client(
        "The answer is DEFINITELY true.",
        "1. The answer is DEFINITELY true.",
        "VERIFIED: True.",
    )
    agent = AntiHallucinator(client, strictness=1.0, content_checks=[UppercaseCheck()])
    result = agent.generate(model="dummy", prompt="p")

    assert len(result.hallucinations_caught) == 1
    assert result.hallucinations_caught[0].startswith(
        "Content check 'no-shouting' flagged:"
    )
    entry = result.verification_log[-1]
    assert entry["content_check"] == "no-shouting"
    assert entry["is_valid"] is False
    summary = result.claim_summary()
    assert summary["flagged_claims"] == 1


def test_clean_content_leaves_no_extra_verdicts() -> None:
    client = _scripted_client(
        "Calm text.", "1. Calm text.", "VERIFIED: True."
    )
    agent = AntiHallucinator(client, strictness=1.0, content_checks=[UppercaseCheck()])
    result = agent.generate(model="dummy", prompt="p")
    assert result.hallucinations_caught == []
    assert result.claim_summary()["total_claims"] == 1


def test_strictness_zero_bypasses_content_checks() -> None:
    client = _scripted_client("LOUD WORDS HERE")
    agent = AntiHallucinator(client, strictness=0.0, content_checks=[UppercaseCheck()])
    result = agent.generate(model="dummy", prompt="p")
    assert result.hallucinations_caught == []
    assert result.verification_log == [{"phase": "bypassed", "reason": "strictness=0.0"}]


def test_failing_check_does_not_break_the_run() -> None:
    class Broken(ContentCheck):
        name = "broken"

        def check(self, content):
            raise RuntimeError("down")

    client = _scripted_client("text", "1. text", "VERIFIED: True.")
    agent = AntiHallucinator(client, strictness=1.0, content_checks=[Broken()])
    result = agent.generate(model="dummy", prompt="p")
    assert result.hallucinations_caught == []


def test_async_pipeline_applies_content_checks() -> None:
    client = _scripted_client(
        "Answer: TRUE WORDS.",
        "1. Answer: TRUE WORDS.",
        "VERIFIED: True.",
    )
    agent = AntiHallucinator(client, strictness=1.0, content_checks=[UppercaseCheck()])
    result = asyncio.run(agent.generate_async(model="dummy", prompt="p"))
    assert any(
        entry.get("content_check") == "no-shouting"
        for entry in result.verification_log
    )


# ------------------------------------------------------------------
# Checks file loading
# ------------------------------------------------------------------

def _write_checks(tmp_path, payload):
    path = tmp_path / "checks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_loader_builds_regex_checks_with_flags(tmp_path) -> None:
    import re

    path = _write_checks(
        tmp_path,
        {
            "checks": [
                {
                    "type": "regex",
                    "name": "no-links",
                    "pattern": r"https?://\S+",
                    "flags": ["IGNORECASE"],
                }
            ]
        },
    )
    checks = load_content_checks(path)
    assert len(checks) == 1
    assert checks[0].name == "no-links"
    assert checks[0].check("See HTTPS://Example.com now") == ["HTTPS://Example.com"]
    assert re.IGNORECASE is not None


def test_loader_requires_a_checks_list(tmp_path) -> None:
    path = _write_checks(tmp_path, {"something": []})
    with pytest.raises(ValueError, match="'checks' list"):
        load_content_checks(path)


def test_loader_rejects_unknown_types_and_flags(tmp_path) -> None:
    path = _write_checks(tmp_path, {"checks": [{"type": "llm", "name": "x"}]})
    with pytest.raises(ValueError, match="unknown content check type"):
        load_content_checks(path)

    path = _write_checks(
        tmp_path, {"checks": [{"type": "regex", "name": "x", "pattern": "y", "flags": ["BOGUS"]}]}
    )
    with pytest.raises(ValueError, match="unknown regex flag"):
        load_content_checks(path)


def test_loader_wraps_read_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot read content checks"):
        load_content_checks(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read content checks"):
        load_content_checks(str(bad))


def test_cli_accepts_checks_on_verify_resume_and_batch() -> None:
    parser = _build_parser()
    args = parser.parse_args(["verify", "--prompt", "p", "--checks", "c.json"])
    assert args.checks == "c.json"
    args = parser.parse_args(["resume", "s.json", "--checks", "c.json"])
    assert args.checks == "c.json"
    args = parser.parse_args(["batch", "--input", "i.jsonl", "--checks", "c.json"])
    assert args.checks == "c.json"
