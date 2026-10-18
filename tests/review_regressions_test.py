"""Integration regressions from the independent PR review."""

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import assets.config as configuration
import benchmark_all as benchmark
import benchmark_knnsampler as coverage
from assets.profiling.line_profile import Profiler
from src.data_preparation.data_preparation import (
    DataGenerator,
    Mar,
    Mcar,
    NoAdditionalMissingValues,
    Rate,
    split_random_state,
)
from src.data_preparation.dataset_description import DatasetDescriptor
from src.imputation import KnnImputer, LinearImputer
from tests.config_test import base_conf, load_from_conf, profile_conf


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
@pytest.mark.parametrize("dataset_type", ["generated", "excel"])
def test_config_splits_seed_and_replays_data(tmp_path, mechanism, dataset_type):
    conf = base_conf(extra="random_state = 42\nMAR_chunk_start = -1\nMAR_chunk_end = 1")
    conf = conf.replace("missing_type = MCAR", f"missing_type = {mechanism}")
    if dataset_type == "excel":
        source = tmp_path / "source.xlsx"
        pd.DataFrame({"X": np.linspace(-2, 2, 120), "Y": range(120)}).to_excel(
            source, index=False
        )
        conf = conf.replace("type = generated", "type = excel")
        conf += f"\npath = {source}\nsheet = Sheet1\ninput = X\ntarget = Y\n"
    config = load_from_conf(tmp_path, conf)
    preparator = config.data_preparator
    assert isinstance(preparator.missing_generator, (Mcar, Mar))
    assert preparator.random_state != preparator.missing_generator.random_state
    first = preparator.prepare_data()
    mask = preparator.missing_indexes.copy()
    second = preparator.prepare_data()
    pd.testing.assert_frame_equal(first.df, second.df)
    np.testing.assert_array_equal(mask, preparator.missing_indexes)
    replay = load_from_conf(tmp_path, conf).data_preparator.prepare_data()
    pd.testing.assert_frame_equal(first.df, replay.df)
    changed = load_from_conf(
        tmp_path, conf.replace("random_state = 42", "random_state = 43")
    )
    assert not first.df.equals(changed.data_preparator.prepare_data().df)


def test_spawned_seeds_are_replayable_and_separate():
    first, second = split_random_state(42)
    assert (first, second) == split_random_state(42)
    assert first != second
    assert (first, second) != split_random_state(43)
    assert split_random_state(None) == (None, None)
    a = np.random.default_rng(first).bit_generator.random_raw(8)
    b = np.random.default_rng(second).bit_generator.random_raw(8)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("kind", [Rate, Mcar, Mar])
def test_hash_survives_mutation_and_subclass_equality(kind):
    rate = Rate(0.3)
    key: Rate | Mcar | Mar
    args: tuple[Any, ...]
    if kind is Rate:
        key = rate
        args = (0.3,)
    elif kind is Mcar:
        key = Mcar(rate, random_state=42)
        args = (Rate(0.3), 42)
    else:
        key = Mar(0, 1, rate, random_state=42)
        args = (0, 1, Rate(0.3), 42)
    subclass = type("Subclass", (kind,), {})(*args)
    assert key == subclass
    assert hash(key) == hash(subclass)
    stored = {key: "present"}
    members = {key}
    original_hash = hash(key)
    rate.nb = 0.7
    if isinstance(key, (Mcar, Mar)):
        key.random_state = 43
        key.missing_values = 5
    if isinstance(key, Mar):
        key.chunk_start = -1
        key.chunk_end = 2
    assert hash(key) == original_hash
    assert stored[key] == "present"
    assert key in members


def test_legacy_constructor_and_attribute_alias(tmp_path):
    descriptor = DatasetDescriptor("feature", "target")
    missing = NoAdditionalMissingValues()
    path = str(tmp_path / "unused.xlsx")
    for generator in (
        DataGenerator(1, 10, missing, "linear", descriptor, path, random_state=42),
        DataGenerator(
            1,
            10,
            missing,
            dataset_descriptor=descriptor,
            excel_path=path,
            random_state=42,
        ),
    ):
        assert generator.dataset_descriptor is descriptor
        assert list(generator.prepare_data().df) == ["feature", "target"]
        assert generator.excel_path == path
        assert not Path(path).exists()
        assert generator.missing_type is missing
        replacement = Mcar(0)
        generator.missing_type = replacement
        assert generator.missing_generator is replacement
        generator.missing_generator = missing
        assert generator.missing_type is missing


def test_legacy_config_loads_lazily_once_and_honors_replacement(monkeypatch):
    # Load a fresh module to avoid polluting the configuration used by other tests.
    name = "assets._legacy_config_test"
    spec = importlib.util.spec_from_file_location(name, configuration.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    with monkeypatch.context() as context:
        context.setattr(
            configuration.configparser.ConfigParser,
            "read",
            lambda *a, **k: pytest.fail("import read config"),
        )
        spec.loader.exec_module(module)
    assert not any(field.name in vars(module) for field in fields(configuration.Config))
    with pytest.raises(AttributeError):
        _ = module.unknown_option
    calls = []
    original = module.load_config

    def load():
        calls.append(True)
        return original()

    monkeypatch.setattr(module, "load_config", load)
    explicit = module.load_config()
    assert "imputers" not in vars(module)
    assert module.imputers
    assert module.imputers is module.imputers
    assert len(calls) == 2  # explicit call and one lazy load
    assert all(field.name in vars(module) for field in fields(configuration.Config))
    assert module.get_field(fields={"present": 3}, field_name="present") == 3
    assert module.get_field({}, "missing") is None
    assert module.mar_chunk == {}
    replacement = DataGenerator(1, 10, NoAdditionalMissingValues(), random_state=43)
    module.__dict__["data_preparator"] = replacement
    prepared, actual = module.prepare_data_from_config()
    assert actual is replacement.actual_values
    pd.testing.assert_frame_equal(prepared.df, replacement.prepare_data().df)
    assert explicit.data_preparator is not replacement
    assert len(calls) == 2


def test_legacy_lazy_load_preserves_early_overrides(monkeypatch):
    name = "assets._early_legacy_config_test"
    spec = importlib.util.spec_from_file_location(name, configuration.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    replacement = DataGenerator(1, 10, NoAdditionalMissingValues(), random_state=42)
    module.__dict__["data_preparator"] = replacement
    module.__dict__["plot_results"] = False
    assert module.imputers
    assert module.data_preparator is replacement
    assert module.plot_results is False
    prepared, _actual = module.prepare_data_from_config()
    assert len(prepared.df) == 10


def coverage_rows():
    rows = []
    for geometry in coverage.geometries:
        for rate in (0.1, 0.3, 0.5):
            for iteration, probability in enumerate((0.8, 1.0)):
                for mechanism in (
                    Mcar(Rate(rate), random_state=iteration),
                    Mar(0.5, 1.5, Rate(rate), random_state=iteration),
                ):
                    for upper in coverage.upper_percentiles:
                        rows.append(
                            (
                                geometry,
                                mechanism,
                                upper,
                                100,
                                rate,
                                iteration,
                                probability,
                            )
                        )
    return rows


def test_coverage_grouping_and_plot_dimensions(monkeypatch):
    monkeypatch.setattr(coverage.plt, "show", lambda: None)
    groups = coverage.transform_results(coverage_rows())
    assert len(groups) == 4
    for _geometry, mechanism, means, stds in groups:
        assert isinstance(mechanism, (Mcar, Mar))
        for mean, std in zip(means, stds, strict=True):
            assert set(mean["Missing Rate"]) == {0.1, 0.3, 0.5}
            np.testing.assert_allclose(mean["Coverage Probability"], 0.9)
            np.testing.assert_allclose(
                std["Coverage Probability"], np.std([0.8, 1.0], ddof=1)
            )
    extra = coverage_rows() + [
        ("linear", Mar(-1, 1, Rate(0.3)), upper, 100, 0.3, iteration, probability)
        for upper in coverage.upper_percentiles
        for iteration, probability in enumerate((0.8, 1.0))
    ]
    more = coverage.transform_results(extra)
    assert len(more) == 5
    try:
        for batch in (groups[:1], groups, more):
            coverage.plot_all_in_one(batch, ["80", "90", "95"])
            assert len(plt.gcf().axes) == len(batch) * len(coverage.upper_percentiles)
            plt.close("all")
    finally:
        plt.close("all")


@pytest.mark.parametrize("per_method", [True, False])
def test_benchmark_preparation_seed_sequence(monkeypatch, per_method):
    monkeypatch.setattr(benchmark, "REGENERATE_DATA_PER_METHOD", per_method)
    monkeypatch.setattr(benchmark.config, "sample_sizes", [20, 30])
    monkeypatch.setattr(
        benchmark, "imputer_classes", {LinearImputer: {}, KnnImputer: {}}
    )
    original = benchmark.create_data_preparator
    configured = benchmark.config.missing_generator
    old_seed = configured.random_state
    captured = []

    def prepare(size, random_state=None):
        preparator = original(size, random_state)
        assert isinstance(preparator.missing_generator, (Mcar, Mar))
        frame = preparator.prepare_data().df
        captured.append(
            (preparator.random_state, preparator.missing_generator.random_state, frame)
        )
        return preparator

    def evaluate(imputers, data, actual):
        return tuple({imputer: 0.1 for imputer in imputers} for _ in range(4))

    monkeypatch.setattr(benchmark, "create_data_preparator", prepare)
    monkeypatch.setattr(benchmark, "evaluate_imputers", evaluate)

    def run(seed):
        captured.clear()
        benchmark.benchmark_for_seed(seed)
        return list(captured)

    first, replay, changed = run(42), run(42), run(43)
    assert len(first) == (4 if per_method else 2)
    assert len({(a, b) for a, b, _ in first}) == len(first)
    for (a, b, frame), (c, d, repeated), (_, _, different) in zip(
        first, replay, changed, strict=True
    ):
        assert a != b
        assert (a, b) == (c, d)
        pd.testing.assert_frame_equal(frame, repeated)
        assert not frame.equals(different)
    assert configured.random_state == old_seed


def test_benchmark_keeps_configured_missingness_seed_without_override(monkeypatch):
    configured = Mar(-1, 1, Rate(0.3), random_state=123)
    monkeypatch.setattr(benchmark.config, "missing_generator", configured)
    preparator = benchmark.create_data_preparator(20)
    assert preparator.missing_generator is not configured
    assert isinstance(preparator.missing_generator, Mar)
    assert preparator.missing_generator.random_state == 123
    assert preparator.missing_generator.chunk_start == -1
    assert preparator.missing_generator.chunk_end == 1
    assert preparator.missing_generator.missing_values == Rate(0.3)
    assert configured.random_state == 123


def test_profile_dump_creates_parent_from_config(tmp_path):
    pytest.importorskip("line_profiler")
    destination = tmp_path / "new" / "nested" / "stats.pkl"
    config = load_from_conf(tmp_path, profile_conf(f'dump_path = "{destination}"'))
    assert config.profile_dump_path is not None
    assert not destination.parent.exists()
    profiler = Profiler([])
    assert profiler.line_profiler is not None
    assert profiler.start_profile(lambda: 7) == 7
    assert profiler.dump_stats(config.profile_dump_path) is profiler
    assert destination.stat().st_size > 0
    profiler.dump_stats(config.profile_dump_path)
    absent = tmp_path / "disabled" / "stats.pkl"
    profiler.line_profiler = None
    profiler.dump_stats(absent)
    assert not absent.parent.exists()
