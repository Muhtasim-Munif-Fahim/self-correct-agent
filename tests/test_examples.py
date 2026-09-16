"""Smoke tests for example scripts."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_demo_script_runs() -> None:
    runpy.run_path(str(Path("examples/demo.py")), run_name="__main__")


def test_research_policy_demo_script_runs(capsys) -> None:
    runpy.run_path(str(Path("examples/research_policy_demo.py")), run_name="__main__")
    out = capsys.readouterr().out
    assert "claims flagged: 1" in out
    assert "Chen et al." in out
    assert "47%" in out
    assert "5 January 2025" in out
