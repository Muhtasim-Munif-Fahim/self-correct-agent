from __future__ import annotations

from types import SimpleNamespace

from self_correct.cli import _build_parser, _verification_exit_code


def test_fail_on_hallucination_returns_nonzero_for_flagged_claims() -> None:
    result = SimpleNamespace(hallucinations_caught=["Flagged claim"])
    assert _verification_exit_code(result, fail_on_hallucination=True) == 1


def test_gate_is_opt_in() -> None:
    result = SimpleNamespace(hallucinations_caught=["Flagged claim"])
    assert _verification_exit_code(result, fail_on_hallucination=False) == 0


def test_clean_verification_passes_with_gate_enabled() -> None:
    result = SimpleNamespace(hallucinations_caught=[])
    assert _verification_exit_code(result, fail_on_hallucination=True) == 0


def test_verify_accepts_persistent_cache_file() -> None:
    args = _build_parser().parse_args(
        ["verify", "--prompt", "Check this", "--cache-file", "claims.json"]
    )
    assert args.cache_file == "claims.json"
