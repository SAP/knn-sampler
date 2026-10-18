"""Verify test initialization in a fresh process, before module collection."""

import os
import shutil
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


def test_reload_refreshes_legacy_configuration_in_fresh_process(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "assets").mkdir()
    shutil.copyfile(root / "assets/config.py", tmp_path / "assets/config.py")
    shutil.copyfile(root / "main.py", tmp_path / "main.py")
    (tmp_path / "assets/hyperparameters.conf").write_text("", encoding="utf-8")
    code = """
import configparser
import importlib
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch
from src.data_preparation.data_preparation import DataGenerator, NoAdditionalMissingValues

path = Path('assets/config.conf')
def write_config(size, plot):
    parser = configparser.ConfigParser()
    parser.read_dict({
        'imputation_algorithms': {'imputation_algorithms': 'LinearImputer'},
        'plot': {'plot_results': str(plot)},
        'profile': {'profile': 'False', 'print_stats': 'False'},
        'dataset_selection': {'used_dataset': 'generated'},
        'generated': {'type': 'generated', 'sample_size': str(size),
                      'linear_interpolation_ratio': '1', 'missing_values': '0',
                      'missing_type': 'MCAR', 'random_state': '42'},
    })
    with path.open('w', encoding='utf-8') as stream:
        parser.write(stream)

write_config(7, True)
with patch.object(configparser.ConfigParser, 'read', side_effect=AssertionError('import read config')):
    import assets.config as config
    import main
assert Path(config.__file__).resolve() == Path('assets/config.py').resolve()
assert config.data_preparator.sample_size == 7
assert main.conf is config
names = {field.name for field in fields(config.Config)}
rows, plots = [], []
def impute(*imputers, **kwargs):
    rows.append(len(imputers[0].ml_data.df))
    return {}
def plot(results, missing, prepared, actual, **kwargs):
    plots.append(len(prepared.df))
main.do_imputation = impute
main.do_plot_results = plot

for size, show in ((12, False), (24, True)):
    config.plot_results = object()  # An explicit reload discards old overrides too.
    write_config(size, show)
    with patch.object(configparser.ConfigParser, 'read', side_effect=AssertionError('reload read config')):
        assert importlib.reload(config) is config
    assert names.isdisjoint(vars(config))
    with patch.object(config, 'load_config', wraps=config.load_config) as loader:
        from assets.config import *
        assert config.plot_results is show
        assert config.data_preparator.sample_size == size
        assert loader.call_count == 1
        from assets.config import *
        with patch.object(configparser.ConfigParser, 'read', side_effect=AssertionError('cached main read config')):
            main.main()
        assert loader.call_count == 1
assert rows == [12, 24]
assert plots == [24]
cached = config.data_preparator
assert config.load_config().data_preparator is not cached
assert config.data_preparator is cached

# Overrides made after reload must survive the subsequent lazy default load.
importlib.reload(config)
replacement = DataGenerator(1, 3, NoAdditionalMissingValues(), random_state=42)
config.data_preparator = replacement
config.plot_results = False
assert config.imputers
assert config.data_preparator is replacement
main.main()
assert rows == [12, 24, 3] and plots == [24]

for failure in (FileNotFoundError, ValueError):
    if failure is FileNotFoundError:
        path.unlink()
    else:
        write_config('invalid', True)
    with patch.object(configparser.ConfigParser, 'read', side_effect=AssertionError('reload read config')):
        importlib.reload(config)
    config.plot_results = False
    try:
        config.imputers
    except failure:
        pass
    else:
        raise AssertionError('invalid configuration was silently reused')
    assert names.intersection(vars(config)) == {'plot_results'}
    write_config(9, True)
    assert config.data_preparator.sample_size == 9
    assert config.plot_results is False
"""
    result = subprocess.run(  # noqa: S603 -- trusted interpreter and constant test code
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root), "MPLBACKEND": "Agg"},
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
