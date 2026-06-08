"""Smoke tests for example scripts."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_demo_script_runs() -> None:
    runpy.run_path(str(Path("examples/demo.py")), run_name="__main__")
