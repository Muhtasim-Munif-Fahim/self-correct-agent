"""Tests for per-phase token usage attribution in verification runs."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from self_correct.core import AntiHallucinationResponse, AntiHallucinator


def _mock(content: str, p: int, c: int) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = p
    resp.usage.completion_tokens = c
    return resp


def _flagged_client():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock("France's capital is Paris. Moon is cheese.", 10, 5),
        _mock("1. France's capital is Paris.\n2. Moon is cheese.", 20, 6),
        _mock("VERIFIED: True.", 30, 7),
        _mock("VERIFIED: False. Moon is rock.", 40, 8),
        _mock("France's capital is Paris.", 50, 9),
    ]
    return client


def _clean_client():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock("Water is wet.", 10, 5),
        _mock("1. Water is wet.", 20, 6),
        _mock("VERIFIED: True.", 30, 7),
    ]
    return client


def test_per_phase_tokens_full_sync_run() -> None:
    result = AntiHallucinator(_flagged_client(), strictness=1.0).generate(
        model="dummy", prompt="Tell me stuff."
    )
    phases = result.token_usage_by_phase
    assert set(phases) == {"drafting", "extraction", "verification", "correction"}
    assert phases["drafting"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert phases["extraction"] == {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26}
    assert phases["verification"] == {"prompt_tokens": 70, "completion_tokens": 15, "total_tokens": 85}
    assert phases["correction"] == {"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59}


def test_per_phase_tokens_sum_to_aggregate() -> None:
    result = AntiHallucinator(_flagged_client(), strictness=1.0).generate(
        model="dummy", prompt="Tell me stuff."
    )
    phases = result.token_usage_by_phase
    assert sum(p["prompt_tokens"] for p in phases.values()) == result.token_usage.prompt_tokens
    assert sum(p["completion_tokens"] for p in phases.values()) == result.token_usage.completion_tokens
    assert sum(p["total_tokens"] for p in phases.values()) == result.token_usage.total_tokens
    assert result.token_usage.total_tokens == 185


def test_per_phase_tokens_bypass_records_drafting_only() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock("Draft.", 10, 5)
    result = AntiHallucinator(client, strictness=0.0).generate(
        model="dummy", prompt="Hello"
    )
    assert set(result.token_usage_by_phase) == {"drafting"}
    assert result.token_usage_by_phase["drafting"] == {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
    }


def test_per_phase_tokens_skip_correction_when_clean() -> None:
    result = AntiHallucinator(_clean_client(), strictness=1.0).generate(
        model="dummy", prompt="Facts?"
    )
    assert set(result.token_usage_by_phase) == {
        "drafting", "extraction", "verification"
    }
    assert "correction" not in result.token_usage_by_phase
    total = sum(p["total_tokens"] for p in result.token_usage_by_phase.values())
    assert total == result.token_usage.total_tokens == 78


def test_per_phase_tokens_async_run() -> None:
    result = asyncio.run(
        AntiHallucinator(_flagged_client(), strictness=1.0).generate_async(
            model="dummy", prompt="Tell me stuff."
        )
    )
    phases = result.token_usage_by_phase
    assert set(phases) == {"drafting", "extraction", "verification", "correction"}
    assert phases["drafting"]["total_tokens"] == 15
    assert phases["verification"]["total_tokens"] == 85
    assert phases["correction"]["total_tokens"] == 59
    assert sum(p["total_tokens"] for p in phases.values()) == result.token_usage.total_tokens


def test_per_phase_tokens_serialized_in_to_dict() -> None:
    result = AntiHallucinator(_flagged_client(), strictness=1.0).generate(
        model="dummy", prompt="p"
    )
    payload = result.to_dict()
    assert payload["token_usage_by_phase"] == result.token_usage_by_phase
    assert payload["token_usage_by_phase"]["verification"]["total_tokens"] == 85


def test_to_json_round_trips_per_phase_tokens() -> None:
    result = AntiHallucinator(_clean_client(), strictness=1.0).generate(
        model="dummy", prompt="p"
    )
    rebuilt = AntiHallucinationResponse.from_dict(json.loads(result.to_json()))
    assert rebuilt.token_usage_by_phase == result.token_usage_by_phase
    assert set(rebuilt.token_usage_by_phase) == {
        "drafting", "extraction", "verification"
    }


def test_from_dict_drops_malformed_phase_tokens() -> None:
    data = {
        "content": "x",
        "verification_log": [],
        "token_usage_by_phase": {
            "drafting": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "bogus": [1, 2, 3],
            "partial": {"prompt_tokens": 1},
        },
    }
    rebuilt = AntiHallucinationResponse.from_dict(data)
    assert set(rebuilt.token_usage_by_phase) == {"drafting"}


def test_default_response_has_empty_per_phase_tokens() -> None:
    assert AntiHallucinationResponse(content="text").token_usage_by_phase == {}


def test_markdown_omits_per_phase_tokens_when_absent() -> None:
    markdown = AntiHallucinationResponse(content="text").to_markdown()
    assert "Per-phase tokens" not in markdown


def test_markdown_renders_per_phase_tokens() -> None:
    result = AntiHallucinationResponse(
        content="text",
        token_usage_by_phase={
            "drafting": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "verification": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        },
    )
    markdown = result.to_markdown()
    assert "Per-phase tokens" in markdown
    assert "drafting 15t (10p/5c)" in markdown
    assert "verification 9t (7p/2c)" in markdown


def test_html_renders_per_phase_tokens_row() -> None:
    result = AntiHallucinationResponse(
        content="text",
        token_usage_by_phase={
            "drafting": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )
    html = result.to_html()
    assert "Per-phase tokens" in html
    assert "drafting 15t (10p/5c)" in html


def test_html_omits_per_phase_tokens_when_absent() -> None:
    html = AntiHallucinationResponse(content="text").to_html()
    assert "Per-phase tokens" not in html
