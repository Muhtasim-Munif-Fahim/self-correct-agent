"""Tests for evidence source domain diversity on AntiHallucinationResponse."""

from __future__ import annotations

import json

from self_correct.core import AntiHallucinationResponse


def _resp(log):
    return AntiHallucinationResponse(content="text", verification_log=list(log))


def test_empty_log_reports_zeroed_counters() -> None:
    div = _resp([]).evidence_diversity()
    assert div == {
        "unique_domains": 0,
        "domain_diversity_ratio": 0.0,
        "domains": [],
        "top_domains": [],
    }


def test_single_source_single_domain_has_full_ratio() -> None:
    div = _resp([
        {"claim": "a", "evidence_sources": [{"url": "https://example.com"}]},
    ]).evidence_diversity()
    assert div["unique_domains"] == 1
    assert div["domain_diversity_ratio"] == 1.0
    assert div["domains"] == ["example.com"]
    assert div["top_domains"] == [{"domain": "example.com", "count": 1}]


def test_sources_on_same_domain_lower_the_ratio() -> None:
    log = [
        {"claim": "a", "evidence_sources": [{"url": "https://example.com/1"}]},
        {"claim": "a", "evidence_sources": [{"url": "https://example.com/2"}]},
        {"claim": "a", "evidence_sources": [{"url": "https://example.com/3"}]},
    ]
    div = _resp(log).evidence_diversity()
    assert div["unique_domains"] == 1
    assert div["domain_diversity_ratio"] == round(1 / 3, 3)
    assert div["top_domains"] == [{"domain": "example.com", "count": 3}]


def test_distinct_domains_maximize_the_ratio() -> None:
    log = [
        {"claim": "a", "evidence_sources": [{"url": "https://a.com"}]},
        {"claim": "a", "evidence_sources": [{"url": "https://b.org"}]},
        {"claim": "a", "evidence_sources": [{"url": "https://c.net"}]},
    ]
    div = _resp(log).evidence_diversity()
    assert div["unique_domains"] == 3
    assert div["domain_diversity_ratio"] == 1.0


def test_mixed_domains_rank_top_domains_by_count_then_name() -> None:
    log = [
        {"claim": "a", "evidence_sources": [{"url": "https://z.com"}]},
        {"claim": "b", "evidence_sources": [
            {"url": "https://a.com"}, {"url": "https://a.com"},
        ]},
        {"claim": "c", "evidence_sources": [{"url": "https://m.io"}]},
        {"claim": "d", "evidence_sources": [{"url": "https://a.com"}]},
    ]
    div = _resp(log).evidence_diversity()
    assert div["unique_domains"] == 3
    assert div["domains"] == ["a.com", "m.io", "z.com"]
    assert div["top_domains"][0] == {"domain": "a.com", "count": 3}
    total = sum(item["count"] for item in div["top_domains"])
    assert total == 5


def test_phase_entries_and_non_dict_sources_are_ignored() -> None:
    log = [
        {"phase": "extraction", "warning": "no claims"},
        {"claim": "a", "evidence_sources": "not-a-list"},
        {"claim": "b", "evidence_sources": [
            {"title": "no url"}, {"url": ""}, {"url": "not a url"},
        ]},
        {"claim": "c", "evidence_sources": [{"url": "https://real.org"}]},
    ]
    div = _resp(log).evidence_diversity()
    assert div["unique_domains"] == 1
    assert div["domains"] == ["real.org"]
    assert div["domain_diversity_ratio"] == 1.0


def test_subdomains_are_treated_as_distinct_hosts() -> None:
    log = [
        {"claim": "a", "evidence_sources": [{"url": "https://news.example.com"}]},
        {"claim": "a", "evidence_sources": [{"url": "https://shop.example.com"}]},
        {"claim": "a", "evidence_sources": [{"url": "https://example.com"}]},
    ]
    div = _resp(log).evidence_diversity()
    assert div["unique_domains"] == 3
    assert div["domain_diversity_ratio"] == 1.0


def test_evidence_diversity_appears_in_to_dict() -> None:
    response = _resp([
        {"claim": "a", "evidence_sources": [{"url": "https://a.com"}]},
        {"claim": "b", "evidence_sources": [{"url": "https://a.com"}]},
    ])
    payload = response.to_dict()
    assert "evidence_diversity" in payload
    assert payload["evidence_diversity"]["unique_domains"] == 1
    assert payload["evidence_diversity"]["domain_diversity_ratio"] == 0.5


def test_evidence_diversity_is_recomputed_from_log() -> None:
    """from_dict recomputes the metric, so the payload round-trips equal."""
    response = _resp([
        {"claim": "a", "evidence_sources": [{"url": "https://a.com"}]},
        {"claim": "b", "evidence_sources": [{"url": "https://b.org"}]},
    ])
    rebuilt = AntiHallucinationResponse.from_dict(response.to_dict())
    assert rebuilt.evidence_diversity() == response.evidence_diversity()


def test_markdown_reports_diversity_when_sources_exist() -> None:
    response = _resp([
        {"claim": "a", "evidence_sources": [{"url": "https://a.com"}]},
        {"claim": "b", "evidence_sources": [{"url": "https://b.org"}]},
    ])
    markdown = response.to_markdown()
    assert "Evidence domain diversity" in markdown
    assert "2 distinct domain(s)" in markdown


def test_markdown_omits_diversity_when_no_sources() -> None:
    markdown = _resp([]).to_markdown()
    assert "Evidence domain diversity" not in markdown


def test_to_json_serializes_diversity() -> None:
    response = _resp([
        {"claim": "a", "evidence_sources": [{"url": "https://a.com"}]},
    ])
    payload = json.loads(response.to_json())
    assert payload["evidence_diversity"]["unique_domains"] == 1
    assert payload["evidence_diversity"]["domain_diversity_ratio"] == 1.0
