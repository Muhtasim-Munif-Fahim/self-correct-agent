from __future__ import annotations

from types import SimpleNamespace

from self_correct.cli import _verification_exit_code


def test_fail_on_hallucination_returns_nonzero_for_flagged_claims() -> None:
    result = SimpleNamespace(hallucinations_caught=["Flagged claim"])
    assert _verification_exit_code(result, fail_on_hallucination=True) == 1


def test_gate_is_opt_in() -> None:
    result = SimpleNamespace(hallucinations_caught=["Flagged claim"])
    assert _verification_exit_code(result, fail_on_hallucination=False) == 0


def test_clean_verification_passes_with_gate_enabled() -> None:
    result = SimpleNamespace(hallucinations_caught=[])
    assert _verification_exit_code(result, fail_on_hallucination=True) == 0
