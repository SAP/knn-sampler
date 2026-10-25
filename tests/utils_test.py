from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import (
    SPLIT_RANDOM_STATE_CHILD_COUNT,
    split_random_state,
)
from src.data_preparation.dataset_description import DatasetDescriptor
from src.imputation import Imputer, KnnImputer, KnnSampler, LinearImputer
from src.utils import do_imputation, instantiate_imputers
from tests.testutils import prepare_generated_data


@pytest.fixture(scope="module")
def generated_data() -> DataFrameMLData:
    prepared, _actual_values, _generator = prepare_generated_data()
    return prepared


@pytest.fixture
def complete_data() -> DataFrameMLData:
    return DataFrameMLData(
        pd.DataFrame({"X": np.arange(20.0), "Y": np.arange(20.0)}),
        DatasetDescriptor("X", "Y"),
    )


@pytest.fixture
def sampler_data() -> DataFrameMLData:
    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 60)
    y = np.sin(x) + rng.normal(0.0, 0.05, 60)
    y[40:] = np.nan
    return DataFrameMLData(
        pd.DataFrame({"X": x, "Y": y}),
        DatasetDescriptor("X", "Y"),
    )


class RecordingImputer(Imputer):
    """Record the execution seeds received through do_imputation."""

    def __init__(self, ml_data: DataFrameMLData):
        super().__init__(ml_data)
        self.seeds: list[int | None] = []

    def fit(self) -> None:
        pass

    def _execute(self, random_state: int | None = None) -> pd.DataFrame:
        self.seeds.append(random_state)
        return self.ml_data.df


def make_sampler(data: DataFrameMLData) -> KnnSampler:
    return KnnSampler(
        ml_data=data,
        compute_bounds=False,
        n_neighbors=3,
        strategy="sample",
    )


class TestExecutionSeeds:
    def test_each_iteration_uses_one_shared_independent_seed(
        self, complete_data: DataFrameMLData
    ):
        first = RecordingImputer(complete_data)
        second = RecordingImputer(complete_data)

        do_imputation(first, second, iterations=3, random_state=11)

        assert first.seeds == second.seeds
        assert len(first.seeds) == 3
        assert len(set(first.seeds)) == 3
        assert all(isinstance(seed, int) for seed in first.seeds)

    def test_seed_derivation_is_independent_of_imputer_order_and_count(
        self, complete_data: DataFrameMLData
    ):
        first = RecordingImputer(complete_data)
        second = RecordingImputer(complete_data)
        do_imputation(first, second, iterations=2, random_state=5)

        alone = RecordingImputer(complete_data)
        do_imputation(alone, iterations=2, random_state=5)

        reversed_first = RecordingImputer(complete_data)
        reversed_second = RecordingImputer(complete_data)
        do_imputation(reversed_second, reversed_first, iterations=2, random_state=5)

        assert first.seeds == second.seeds == alone.seeds
        assert reversed_first.seeds == reversed_second.seeds == first.seeds

    def test_without_seed_forwards_none(self, complete_data: DataFrameMLData):
        imputer = RecordingImputer(complete_data)

        do_imputation(imputer, iterations=3)

        assert imputer.seeds == [None, None, None]

    def test_zero_seed_is_not_treated_as_none(self, complete_data: DataFrameMLData):
        imputer = RecordingImputer(complete_data)

        do_imputation(imputer, iterations=2, random_state=0)

        assert len(imputer.seeds) == 2
        assert all(seed is not None for seed in imputer.seeds)

    def test_single_run_matches_first_seeded_iteration(
        self, complete_data: DataFrameMLData
    ):
        single = RecordingImputer(complete_data)
        do_imputation(single, random_state=7)

        multiple = RecordingImputer(complete_data)
        do_imputation(multiple, iterations=3, random_state=7)

        assert single.seeds[0] == multiple.seeds[0]

    @pytest.mark.parametrize("random_state", [-1, 1.5, "42", True, np.bool_(True)])
    def test_invalid_seed_raises(self, complete_data: DataFrameMLData, random_state):
        imputer = RecordingImputer(complete_data)

        with pytest.raises(ValueError, match=r"random_state.*non-negative integer"):
            do_imputation(imputer, random_state=random_state)

    def test_numpy_integer_seed_matches_python_integer(
        self, complete_data: DataFrameMLData
    ):
        numpy_seeded = RecordingImputer(complete_data)
        do_imputation(numpy_seeded, iterations=2, random_state=cast(int, np.int64(7)))

        python_seeded = RecordingImputer(complete_data)
        do_imputation(python_seeded, iterations=2, random_state=7)

        assert numpy_seeded.seeds == python_seeded.seeds

    @pytest.mark.parametrize(
        ("execution_seed", "dataset_seed"),
        [
            (42, 42),
            # Large execution and dataset seeds exercise multi-word entropy
            # and 32-bit truncation; derived seeds must remain disjoint.
            (42, 2**32 + 42),
            (0, 2**32),
            (42, 42 + (2 << 128)),
            (42 + (2 << 128), 42 + (2 << 128) + 2**160),
            (2**32 + 42, 2**64 + 2**32 + 42),
        ],
    )
    def test_execution_seeds_are_disjoint_from_preparation_seeds(
        self,
        complete_data: DataFrameMLData,
        execution_seed: int,
        dataset_seed: int,
    ):
        preparation_seed, missingness_seed = split_random_state(dataset_seed)
        reserved_seeds = {
            int(child.generate_state(1)[0])
            for child in np.random.SeedSequence(dataset_seed).spawn(
                SPLIT_RANDOM_STATE_CHILD_COUNT
            )
        }
        imputer = RecordingImputer(complete_data)

        do_imputation(imputer, iterations=5, random_state=execution_seed)

        assert len(set(imputer.seeds)) == 5
        assert {preparation_seed, missingness_seed} <= reserved_seeds
        assert set(imputer.seeds).isdisjoint(reserved_seeds)


class TestSeededKnnSampler:
    def test_same_seed_is_reproducible(self, sampler_data: DataFrameMLData):
        first = make_sampler(sampler_data)
        second = make_sampler(sampler_data)

        first_result = do_imputation(first, random_state=7)[first][0]
        second_result = do_imputation(second, random_state=7)[second][0]

        pd.testing.assert_frame_equal(first_result, second_result)

    def test_different_seeds_differ(self, sampler_data: DataFrameMLData):
        first = make_sampler(sampler_data)
        second = make_sampler(sampler_data)

        first_result = do_imputation(first, random_state=7)[first][0]
        second_result = do_imputation(second, random_state=8)[second][0]

        assert not first_result.equals(second_result)

    def test_single_run_matches_first_seeded_iteration(
        self, sampler_data: DataFrameMLData
    ):
        single = make_sampler(sampler_data)
        single_result = do_imputation(single, random_state=7)[single][0]

        multiple = make_sampler(sampler_data)
        multiple_results = do_imputation(multiple, iterations=3, random_state=7)[
            multiple
        ]
        first_iteration = multiple_results[0][0]

        pd.testing.assert_frame_equal(single_result, first_iteration)

    def test_seeded_iterations_are_distinct_and_reproducible(
        self, sampler_data: DataFrameMLData
    ):
        first = make_sampler(sampler_data)
        second = make_sampler(sampler_data)

        first_results = do_imputation(first, iterations=3, random_state=7)[first]
        second_results = do_imputation(second, iterations=3, random_state=7)[second]

        first_frames = [frame for frame, _elapsed in first_results]
        second_frames = [frame for frame, _elapsed in second_results]
        assert len(first_frames) == 3
        assert len({frame.to_numpy().tobytes() for frame in first_frames}) == 3
        for first_frame, second_frame in zip(first_frames, second_frames, strict=True):
            pd.testing.assert_frame_equal(first_frame, second_frame)


class TestDoImputation:
    def test_single_iteration_returns_one_timed_result(
        self, generated_data: DataFrameMLData
    ):
        imputer = KnnImputer(ml_data=generated_data)
        results = do_imputation(imputer)
        assert list(results.keys()) == [imputer]
        result, execution_time = results[imputer]
        assert isinstance(result, pd.DataFrame)
        assert execution_time >= 0.0
        assert not result["Y"].isna().any()

    def test_iterations_return_one_result_per_iteration(
        self, generated_data: DataFrameMLData
    ):
        imputer = KnnImputer(ml_data=generated_data)
        iterations = 3
        results = do_imputation(imputer, iterations=iterations)
        assert list(results.keys()) == [imputer]
        assert len(results[imputer]) == iterations
        for result, execution_time in results[imputer]:
            assert isinstance(result, pd.DataFrame)
            assert execution_time >= 0.0


class TestInstantiateImputers:
    def test_instantiates_with_kwargs(self, generated_data: DataFrameMLData):
        imputers = list(
            instantiate_imputers({KnnImputer: {"n_neighbors": 3}}, generated_data)
        )
        assert len(imputers) == 1
        assert isinstance(imputers[0], KnnImputer)
        assert imputers[0].knn_imputer.n_neighbors == 3

    def test_unknown_kwarg_raises_on_iteration(self, generated_data: DataFrameMLData):
        # instantiate_imputers is a generator: the error fires on iteration
        generator = instantiate_imputers(
            {LinearImputer: {"bogus_option": 1}}, generated_data
        )
        with pytest.raises(ValueError, match="bogus_option"):
            list(generator)
