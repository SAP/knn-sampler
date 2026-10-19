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
    monkeypatch, tmp_path, loaded, plot, profile, dump, print_stats
):
    # Restore the complete module namespace, including attributes loaded lazily.
    with patch.dict(configuration.__dict__):
        for field in fields(configuration.Config):
            configuration.__dict__.pop(field.name, None)
        if loaded:
            assert configuration.imputers

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
        assert calls["kwargs"] == ({"iterations": None} if profile else {})
        assert calls.get("profile", False) == profile
        assert calls.get("plot", False) == plot
        assert calls.get("dump") == destination
        assert calls.get("print", False) == print_stats
        assert not (tmp_path / "not_written.pkl").exists()


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
