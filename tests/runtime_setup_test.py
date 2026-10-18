"""Verify test initialization in a fresh process, before module collection."""

import os
import subprocess
import sys
from pathlib import Path


def test_pytest_forces_headless_backend_before_collection() -> None:
    code = """
import matplotlib
import pytest

class CheckBackend:
    def pytest_sessionstart(self, session):
        assert matplotlib.get_backend().lower() == 'agg', matplotlib.get_backend()

raise SystemExit(pytest.main(
    ['--collect-only', '-q', 'tests/stats_utils_test.py'],
    plugins=[CheckBackend()],
))
"""
    result = subprocess.run(  # noqa: S603 -- trusted interpreter and constant test code
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MPLBACKEND": "TkAgg"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
