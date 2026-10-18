"""Verify dataset seeding under both benchmark regeneration policies."""

import pandas as pd
import pytest

import benchmark_all as benchmark
from src.data_preparation.data_preparation import Mar, Mcar, Rate
from src.imputation import KnnImputer, LinearImputer


@pytest.mark.parametrize("per_method", [True, False])
@pytest.mark.parametrize("root_seed", [0, 42])
def test_benchmark_preparation_seed_sequence(monkeypatch, per_method, root_seed):
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

    first, replay, changed = run(root_seed), run(root_seed), run(root_seed + 1)
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


@pytest.mark.parametrize("per_method", [True, False])
def test_benchmark_caller_without_root_preserves_configured_seed(
    monkeypatch, per_method
):
    configured = Mar(-1, 1, Rate(0.3), random_state=123)
    monkeypatch.setattr(benchmark.config, "missing_generator", configured)
    monkeypatch.setattr(benchmark.config, "sample_sizes", [20, 30])
    monkeypatch.setattr(benchmark, "REGENERATE_DATA_PER_METHOD", per_method)
    monkeypatch.setattr(
        benchmark, "imputer_classes", {LinearImputer: {}, KnnImputer: {}}
    )
    original = benchmark.create_data_preparator
    captured = []

    def prepare(size, random_state=None):
        preparator = original(size, random_state)
        captured.append((random_state, preparator))
        return preparator

    def evaluate(imputers, data, actual):
        return tuple({imputer: 0.1 for imputer in imputers} for _ in range(4))

    monkeypatch.setattr(benchmark, "create_data_preparator", prepare)
    monkeypatch.setattr(benchmark, "evaluate_imputers", evaluate)
    benchmark.benchmark_for_seed()
    assert len(captured) == (4 if per_method else 2)
    for passed_seed, preparator in captured:
        assert passed_seed is None
        assert preparator.random_state is None
        assert isinstance(preparator.missing_generator, Mar)
        assert preparator.missing_generator is not configured
        assert preparator.missing_generator == configured
    assert configured.random_state == 123
