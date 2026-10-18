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


def test_legacy_star_import_in_fresh_process() -> None:
    code = """
from unittest.mock import patch
import configparser
import pandas

with patch.object(configparser.ConfigParser, 'read', side_effect=AssertionError('import read config')), \
     patch.object(pandas, 'read_excel', side_effect=AssertionError('import read data')):
    import assets.config as config
    import main
    from assets.config import Config, load_config, get_field
    legacy = {'imputers', 'plot_results', 'profile', 'print_profile_stats',
              'profile_dump_path', 'data_preparator', 'missing_values'}
    assert legacy.isdisjoint(vars(config))
    public = legacy | {'DEFAULT_CONFIG_PATH', 'DEFAULT_HYPERPARAMETERS_PATH',
                       'PROJECT_ROOT', 'Config', 'get_field', 'load_config',
                       'mar_chunk', 'prepare_data_from_config'}
    assert set(config.__all__) == public
    assert len(config.__all__) == len(public)

override = object()
config.plot_results = override
sentinel = object()
fields = Path = pd = sentinel
with patch.object(config, 'load_config', wraps=config.load_config) as loader:
    from assets.config import *
    assert legacy <= globals().keys(), legacy - globals().keys()
    assert loader.call_count == 1
    assert plot_results is override
    assert config.plot_results is override
    assert fields is Path is pd is sentinel
    assert get_field({'x': 1}, 'x') == 1
    assert mar_chunk == {}
    assert callable(prepare_data_from_config)
    from assets.config import *
    assert loader.call_count == 1
"""
    result = subprocess.run(  # noqa: S603 -- trusted interpreter and constant test code
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
