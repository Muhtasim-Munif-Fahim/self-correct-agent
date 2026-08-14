"""Tests for the persistent run history."""

import json

from self_correct import history


class TestRecordAndLoad:
    def test_round_trips_a_run(self, tmp_path):
        path = tmp_path / "history.jsonl"
        history.record_run({"command": "verify", "model": "gpt-4o-mini"}, path=path)

        runs = history.load_runs(path)
        assert len(runs) == 1
        assert runs[0]["model"] == "gpt-4o-mini"
        assert "timestamp" in runs[0], "every record is stamped"

    def test_appends_rather_than_overwrites(self, tmp_path):
        path = tmp_path / "history.jsonl"
        for i in range(3):
            history.record_run({"command": "verify", "model": f"m{i}"}, path=path)
        assert [r["model"] for r in history.load_runs(path)] == ["m0", "m1", "m2"]

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert history.load_runs(tmp_path / "absent.jsonl") == []

    def test_truncated_final_line_does_not_break_reading(self, tmp_path):
        """An interrupted write must not make the whole history unreadable."""
        path = tmp_path / "history.jsonl"
        history.record_run({"command": "verify", "model": "good"}, path=path)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"command": "verify", "mod')

        runs = history.load_runs(path)
        assert [r["model"] for r in runs] == ["good"]

    def test_unwritable_path_does_not_raise(self, tmp_path):
        """History is a convenience; it must never fail the actual run."""
        path = tmp_path / "nope"
        path.mkdir()
        history.record_run({"command": "verify"}, path=path)  # a directory

    def test_trims_to_the_most_recent_records(self, tmp_path, monkeypatch):
        path = tmp_path / "history.jsonl"
        monkeypatch.setattr(history, "MAX_RECORDS", 5)
        for i in range(9):
            history.record_run({"command": "verify", "model": f"m{i}"}, path=path)

        runs = history.load_runs(path)
        assert len(runs) == 5
        assert [r["model"] for r in runs] == [f"m{i}" for i in range(4, 9)]


class TestAggregate:
    def test_empty(self):
        assert history.aggregate([]) == {"runs": 0}

    def test_totals_and_rates(self):
        runs = [
            {"timestamp": 10, "model": "a", "claims": 4, "claims_verified": 2,
             "duration": 2.0, "prompt_tokens": 100, "cache_hits": 1, "cache_misses": 3},
            {"timestamp": 20, "model": "a", "claims": 6, "claims_verified": 6,
             "duration": 4.0, "prompt_tokens": 50, "cache_hits": 5, "cache_misses": 1},
        ]
        summary = history.aggregate(runs)

        assert summary["runs"] == 2
        assert summary["claims"] == 10
        assert summary["claims_verified"] == 8
        assert summary["verified_rate"] == 0.8
        assert summary["prompt_tokens"] == 150
        assert summary["cache_hits"] == 6
        assert summary["mean_duration"] == 3.0
        assert summary["models"] == {"a": 2}

    def test_counts_errors_and_tolerates_missing_fields(self):
        summary = history.aggregate([
            {"timestamp": 1, "model": "a", "error": "boom"},
            {"timestamp": 2, "model": "b"},
        ])
        assert summary["runs"] == 2
        assert summary["errors"] == 1
        assert summary["claims"] == 0
        assert summary["verified_rate"] == 0.0


class TestHistoryPath:
    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv(history.HISTORY_PATH_ENV, str(tmp_path / "x.jsonl"))
        assert history.history_path() == tmp_path / "x.jsonl"
