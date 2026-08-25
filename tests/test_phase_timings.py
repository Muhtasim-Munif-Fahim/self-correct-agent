"""Tests for per-phase timing capture in verification runs."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from self_correct.core import AntiHallucinationResponse, AntiHallucinator


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def _flagged_client() -> MagicMock:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("France's capital is Paris. Moon is cheese."),
        _mock_response("1. France's capital is Paris.\n2. Moon is cheese."),
        _mock_response("VERIFIED: True."),
        _mock_response("VERIFIED: False. Moon is rock."),
        _mock_response("France's capital is Paris."),
    ]
    return mock_client


def _clean_client() -> MagicMock:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("Water is wet."),
        _mock_response("1. Water is wet."),
        _mock_response("VERIFIED: True."),
    ]
    return mock_client


def test_full_run_records_every_phase() -> None:
    agent = AntiHallucinator(_flagged_client(), strictness=1.0)
    result = agent.generate(model="dummy", prompt="Tell me stuff.")

    assert set(result.phase_timings) == {
        "drafting", "extraction", "verification", "correction",
    }
    for seconds in result.phase_timings.values():
        assert seconds >= 0.0


def test_bypassed_run_records_drafting_only() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response("Draft.")
    agent = AntiHallucinator(mock_client, strictness=0.0)
    result = agent.generate(model="dummy", prompt="Hello")

    assert set(result.phase_timings) == {"drafting"}


def test_clean_run_has_no_correction_timing() -> None:
    agent = AntiHallucinator(_clean_client(), strictness=1.0)
    result = agent.generate(model="dummy", prompt="Facts?")

    assert set(result.phase_timings) == {"drafting", "extraction", "verification"}


def test_budget_exhaustion_stops_after_drafting() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response("Draft.")
    agent = AntiHallucinator(mock_client, strictness=1.0, max_llm_calls=1)
    result = agent.generate(model="dummy", prompt="Hello")

    assert set(result.phase_timings) == {"drafting"}
    assert result.budget_report()["exhausted"] is True


def test_async_run_records_the_same_phase_names() -> None:
    agent = AntiHallucinator(_flagged_client(), strictness=1.0)
    result = asyncio.run(agent.generate_async(model="dummy", prompt="Tell me stuff."))

    assert list(result.phase_timings) == [
        "drafting", "extraction", "verification", "correction",
    ]


def test_timings_appear_in_serialized_output() -> None:
    response = AntiHallucinationResponse(
        content="text",
        phase_timings={"drafting": 0.123456, "verification": 1.0},
    )
    payload = json.loads(response.to_json())
    assert payload["phase_timings"] == {"drafting": 0.123, "verification": 1.0}
    assert "phase_timings" not in AntiHallucinationResponse(content="t").to_dict() or \
        AntiHallucinationResponse(content="t").to_dict()["phase_timings"] == {}


def test_markdown_reports_phase_breakdown() -> None:
    response = AntiHallucinationResponse(
        content="text",
        phase_timings={"drafting": 0.5, "verification": 2.25},
    )
    markdown = response.to_markdown()
    assert "- **Phase timings**: drafting 0.50s, verification 2.25s" in markdown


def test_markdown_omits_breakdown_without_timings() -> None:
    markdown = AntiHallucinationResponse(content="text").to_markdown()
    assert "**Phase timings**" not in markdown


def test_default_response_has_empty_timings() -> None:
    assert AntiHallucinationResponse(content="text").phase_timings == {}
