"""Tests for time-bucketed history trends."""

from __future__ import annotations

import json

import pytest

from self_correct import history
from self_correct.cli import main

DAY = 86400


def _run(timestamp: float, *, claims: int = 10, verified: int = 9, duration: float = 1.0, **extra):
    return {
        "timestamp": timestamp,
        "claims": claims,
        "claims_verified": verified,
        "duration": duration,
        "model": "test-model",
        **extra,
    }


class TestTrendBuckets:
    def test_groups_runs_by_day(self) -> None:
        runs = [_run(0), _run(100), _run(DAY), _run(2 * DAY)]
        buckets = history.trend_buckets(runs)
        assert [b["runs"] for b in buckets] == [2, 1, 1]

    def test_buckets_are_epoch_aligned_not_first_run_aligned(self) -> None:
        # Two runs an hour apart either side of midnight must land in
        # different day buckets.
        buckets = history.trend_buckets([_run(DAY - 1800), _run(DAY + 1800)])
        assert len(buckets) == 2

    def test_ordered_oldest_first(self) -> None:
        buckets = history.trend_buckets([_run(3 * DAY), _run(0), _run(DAY)])
        starts = [b["bucket_start"] for b in buckets]
        assert starts == sorted(starts)

    def test_gaps_produce_no_empty_buckets(self) -> None:
        buckets = history.trend_buckets([_run(0), _run(10 * DAY)])
        assert len(buckets) == 2

    def test_min_runs_drops_thin_buckets(self) -> None:
        runs = [_run(0), _run(100), _run(DAY)]
        buckets = history.trend_buckets(runs, min_runs=2)
        assert [b["runs"] for b in buckets] == [2]

    def test_hour_and_week_widths(self) -> None:
        runs = [_run(0), _run(7200)]
        assert len(history.trend_buckets(runs, bucket="hour")) == 2
        assert len(history.trend_buckets(runs, bucket="week")) == 1

    def test_verified_rate_is_computed_per_bucket(self) -> None:
        runs = [_run(0, claims=10, verified=5), _run(DAY, claims=10, verified=10)]
        rates = [b["verified_rate"] for b in history.trend_buckets(runs)]
        assert rates == [0.5, 1.0]

    def test_unparseable_timestamp_is_skipped(self) -> None:
        runs = [_run(0), {"timestamp": "not-a-number", "claims": 1}]
        assert sum(b["runs"] for b in history.trend_buckets(runs)) == 1

    def test_no_runs_returns_no_buckets(self) -> None:
        assert history.trend_buckets([]) == []

    def test_bucket_must_be_known(self) -> None:
        with pytest.raises(ValueError, match="bucket must be one of"):
            history.trend_buckets([], bucket="fortnight")

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True])
    def test_min_runs_validation(self, bad: object) -> None:
        with pytest.raises(ValueError, match="min_runs must be a positive integer"):
            history.trend_buckets([], min_runs=bad)


class TestTrendDirection:
    def _buckets(self, rates):
        return [{"verified_rate": r, "mean_duration": 1.0} for r in rates]

    def test_falling_rate_is_declining(self) -> None:
        result = history.trend_direction(self._buckets([0.9, 0.9, 0.4, 0.4]))
        assert result["verdict"] == "declining"
        assert result["change"] == pytest.approx(-0.5)

    def test_rising_rate_is_improving(self) -> None:
        assert history.trend_direction(self._buckets([0.4, 0.4, 0.9, 0.9]))["verdict"] == "improving"

    def test_small_change_is_stable_within_tolerance(self) -> None:
        assert history.trend_direction(self._buckets([0.90, 0.88]))["verdict"] == "stable"

    def test_tolerance_zero_makes_any_change_a_trend(self) -> None:
        result = history.trend_direction(self._buckets([0.90, 0.88]), tolerance=0.0)
        assert result["verdict"] == "declining"

    def test_odd_bucket_count_excludes_the_middle(self) -> None:
        # The middle bucket must not land in both halves and drag them together.
        result = history.trend_direction(self._buckets([1.0, 0.0, 0.0]))
        assert result["first_half"] == 1.0
        assert result["second_half"] == 0.0

    def test_one_bucket_is_insufficient(self) -> None:
        result = history.trend_direction(self._buckets([0.9]))
        assert result["verdict"] == "insufficient-data"
        assert result["change"] == 0.0

    def test_alternate_metric_is_honoured(self) -> None:
        buckets = [
            {"verified_rate": 0.9, "mean_duration": 1.0},
            {"verified_rate": 0.9, "mean_duration": 9.0},
        ]
        result = history.trend_direction(buckets, metric="mean_duration", tolerance=0.5)
        assert result["metric"] == "mean_duration"
        # Getting slower is a regression even though the number went up.
        assert result["verdict"] == "declining"
        assert result["change"] == pytest.approx(8.0)

    def test_falling_duration_is_an_improvement(self) -> None:
        buckets = [{"mean_duration": 9.0}, {"mean_duration": 1.0}]
        result = history.trend_direction(buckets, metric="mean_duration", tolerance=0.5)
        assert result["verdict"] == "improving"
        assert result["change"] == pytest.approx(-8.0)

    def test_rising_error_count_is_a_regression(self) -> None:
        buckets = [{"errors": 0}, {"errors": 5}]
        assert history.trend_direction(buckets, metric="errors")["verdict"] == "declining"

    def test_lower_is_better_metrics_are_declared(self) -> None:
        assert "mean_duration" in history.LOWER_IS_BETTER
        assert "errors" in history.LOWER_IS_BETTER
        assert "verified_rate" not in history.LOWER_IS_BETTER

    def test_missing_metric_is_treated_as_insufficient(self) -> None:
        result = history.trend_direction(self._buckets([0.9, 0.8]), metric="nope")
        assert result["verdict"] == "insufficient-data"

    @pytest.mark.parametrize("bad", [-0.1, True, "0.1"])
    def test_tolerance_validation(self, bad: object) -> None:
        with pytest.raises(ValueError, match="tolerance must be a non-negative number"):
            history.trend_direction([], tolerance=bad)


class TestTrendCli:
    @staticmethod
    def _history(tmp_path, monkeypatch, runs) -> None:
        path = tmp_path / "history.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in runs), encoding="utf-8")
        monkeypatch.setenv(history.HISTORY_PATH_ENV, str(path))

    def test_table_reports_the_verdict(self, tmp_path, monkeypatch, capsys) -> None:
        self._history(
            tmp_path, monkeypatch,
            [_run(0, verified=9), _run(DAY, verified=9), _run(2 * DAY, verified=2), _run(3 * DAY, verified=2)],
        )
        assert main(["trend"]) == 0
        assert "declining" in capsys.readouterr().out

    def test_json_output_carries_buckets_and_trend(self, tmp_path, monkeypatch, capsys) -> None:
        self._history(tmp_path, monkeypatch, [_run(0), _run(DAY)])
        assert main(["trend", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["buckets"]) == 2
        assert payload["trend"]["metric"] == "verified_rate"

    def test_empty_history_reports_cleanly(self, tmp_path, monkeypatch, capsys) -> None:
        self._history(tmp_path, monkeypatch, [])
        assert main(["trend"]) == 0
        assert "No runs recorded yet." in capsys.readouterr().out

    def test_single_bucket_explains_insufficient_data(self, tmp_path, monkeypatch, capsys) -> None:
        self._history(tmp_path, monkeypatch, [_run(0), _run(50)])
        assert main(["trend"]) == 0
        assert "Need at least 2 buckets" in capsys.readouterr().out

    def test_invalid_min_runs_exits_nonzero(self, tmp_path, monkeypatch, capsys) -> None:
        self._history(tmp_path, monkeypatch, [_run(0)])
        assert main(["trend", "--min-runs", "0"]) == 1
        assert "--min-runs must be a positive integer" in capsys.readouterr().out

    def test_invalid_tolerance_exits_nonzero(self, tmp_path, monkeypatch, capsys) -> None:
        self._history(tmp_path, monkeypatch, [_run(0)])
        assert main(["trend", "--tolerance", "-1"]) == 1
        assert "--tolerance must be a non-negative number" in capsys.readouterr().out
