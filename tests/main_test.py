"""Exercise the application entry point with historical Python configuration."""

from dataclasses import fields
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

import assets.config as configuration
import main as entrypoint
from src.data_preparation.data_preparation import (
    DataGenerator,
    ExcelDataPreparator,
    NoAdditionalMissingValues,
)
from src.data_preparation.dataset_description import ExcelDatasetDescriptor
from src.imputation import LinearImputer
from tests.config_test import base_conf, load_from_conf


@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize("plot", [False, True])
@pytest.mark.parametrize("execution_seed", [pytest.param(None, id="unset"), 0, 17])
@pytest.mark.parametrize(
    "profile,dump,print_stats",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
    ],
)
def test_main_honors_module_overrides(
    monkeypatch, tmp_path, loaded, plot, execution_seed, profile, dump, print_stats
):
    # Restore the complete module namespace, including attributes loaded lazily.
    with patch.dict(configuration.__dict__):
        for field in fields(configuration.Config):
            configuration.__dict__.pop(field.name, None)
        if loaded:
            assert configuration.imputers
            # Historical callers never set settings added to Config later.
            configuration.__dict__.pop("execution_random_state", None)

        preparator = DataGenerator(1, 7, NoAdditionalMissingValues(), random_state=7)
        destination = tmp_path / "not_written.pkl" if dump else None
        overrides = dict(
            data_preparator=preparator,
            imputers={LinearImputer: {}},
            missing_values=0,
            profile=profile,
            plot_results=plot,
            profile_dump_path=destination,
            print_profile_stats=print_stats,
        )
        if execution_seed is not None:
            overrides["execution_random_state"] = execution_seed
        configuration.__dict__.update(overrides)
        calls: dict[str, Any] = {}

        def impute(*imputers, **kwargs):
            assert len(imputers) == 1
            imputer = imputers[0]
            assert isinstance(imputer, LinearImputer)
            calls["prepared"] = imputer.ml_data
            calls["kwargs"] = kwargs
            calls["results"] = {imputer: (imputer.ml_data.df, 0.1)}
            return calls["results"]

        def plots(results, missing, prepared, actual, *, sample_size):
            assert results is calls["results"]
            assert missing == 0
            assert prepared is calls["prepared"]
            assert actual is preparator.actual_values
            assert sample_size == 7
            calls["plot"] = True

        class SpyProfiler:
            def __init__(self, functions):
                assert functions == [LinearImputer._execute]
                calls["profile"] = True

            def start_profile(self, function, *args, **kwargs):
                assert function is impute
                return function(*args, **kwargs)

            def dump_stats(self, path):
                calls["dump"] = path

            def print_stats(self):
                calls["print"] = True

        monkeypatch.setattr(entrypoint, "do_imputation", impute)
        monkeypatch.setattr(entrypoint, "do_plot_results", plots)
        monkeypatch.setattr(entrypoint, "Profiler", SpyProfiler)
        monkeypatch.setattr(
            configuration.configparser.ConfigParser,
            "read",
            lambda *a, **k: pytest.fail(
                "main reread configuration instead of using overrides"
            ),
        )
        entrypoint.main()
        assert calls["prepared"].df.equals(preparator.prepare_data().df)
        expected_kwargs: dict[str, int | None] = {"random_state": execution_seed}
        if profile:
            expected_kwargs["iterations"] = None
        assert calls["kwargs"] == expected_kwargs
        assert calls.get("profile", False) == profile
        assert calls.get("plot", False) == plot
        assert calls.get("dump") == destination
        assert calls.get("print", False) == print_stats
        assert not (tmp_path / "not_written.pkl").exists()


@pytest.mark.parametrize("profile", [False, True])
def test_main_reads_file_seed_for_partial_overrides(monkeypatch, tmp_path, profile):
    """An incomplete override set still resolves unspecified settings from files."""
    config_path = tmp_path / "config.conf"
    config_path.write_text(
        base_conf(
            imputers="LinearImputer",
            extra="\n[execution]\nexecution_random_state = 23",
        ),
        encoding="utf-8",
    )
    hyperparameters_path = tmp_path / "hyperparameters.conf"
    hyperparameters_path.write_text("", encoding="utf-8")
    preparator = DataGenerator(1, 7, NoAdditionalMissingValues(), random_state=7)
    calls: dict[str, Any] = {}

    def impute(*imputers, **kwargs):
        calls["kwargs"] = kwargs
        return {}

    class PassthroughProfiler:
        def __init__(self, functions):
            assert functions == [LinearImputer._execute]

        def start_profile(self, function, *args, **kwargs):
            return function(*args, **kwargs)

        def dump_stats(self, path):
            pass

        def print_stats(self):
            pass

    with patch.dict(configuration.__dict__):
        for field in fields(configuration.Config):
            configuration.__dict__.pop(field.name, None)
        # missing_values is intentionally omitted: the override set is partial,
        # so unspecified settings, including the execution seed, come from files.
        configuration.__dict__.update(
            data_preparator=preparator,
            imputers={LinearImputer: {}},
            plot_results=False,
            profile=profile,
            profile_dump_path=None,
            print_profile_stats=False,
        )
        monkeypatch.setattr(entrypoint, "do_imputation", impute)
        monkeypatch.setattr(entrypoint, "Profiler", PassthroughProfiler)
        monkeypatch.setattr(configuration, "DEFAULT_CONFIG_PATH", config_path)
        monkeypatch.setattr(
            configuration, "DEFAULT_HYPERPARAMETERS_PATH", hyperparameters_path
        )
        entrypoint.main()

    expected_kwargs: dict[str, int | None] = {"random_state": 23}
    if profile:
        expected_kwargs["iterations"] = None
    assert calls["kwargs"] == expected_kwargs


def test_main_reports_observed_size_for_all_rows(tmp_path, monkeypatch):
    import main as entrypoint

    source = tmp_path / "all.xlsx"
    pd.DataFrame({"X": [1.0, 1.0], "Y": [2.0, 3.0]}).to_excel(source, index=False)
    config = load_from_conf(
        tmp_path, base_conf(imputers="LinearImputer"), hyperparameters=""
    )
    config.plot_results = True
    config.data_preparator = ExcelDataPreparator(
        ExcelDatasetDescriptor("X", "Y", str(source), "Sheet1"),
        NoAdditionalMissingValues(),
    )
    monkeypatch.setattr(entrypoint, "conf", config)
    monkeypatch.setattr(entrypoint, "do_imputation", lambda *a, **k: {})
    sizes = []
    monkeypatch.setattr(
        entrypoint, "do_plot_results", lambda *a, **k: sizes.append(k["sample_size"])
    )
    entrypoint.main()
    assert sizes == [2]


@pytest.mark.parametrize("seed", [None, 0])
def test_main_forwards_execution_seed(tmp_path, monkeypatch, seed):
    config = load_from_conf(
        tmp_path, base_conf(imputers="LinearImputer"), hyperparameters=""
    )
    config.plot_results = False
    config.execution_random_state = seed
    config.data_preparator = DataGenerator(
        1, 7, NoAdditionalMissingValues(), random_state=7
    )
    monkeypatch.setattr(entrypoint, "conf", config)
    calls = {}

    def impute(*imputers, **kwargs):
        calls["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(entrypoint, "do_imputation", impute)
    entrypoint.main()

    assert calls["kwargs"] == {"random_state": seed}
