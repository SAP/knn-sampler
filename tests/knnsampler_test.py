import unittest
from typing import Any, ClassVar, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold as SklearnKFold

from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.dataset_description import DatasetDescriptor
from src.imputation.knnsampler import (
    DEFAULT_RANDOM_STATE,
    MAX_KFOLD_RANDOM_STATE,
    KnnSampler,
)


def make_ml_data(
    total_size: int,
    complete_size: int,
    *,
    missing_input_index: int | None = None,
    complete_input_index: int | None = None,
) -> DataFrameMLData:
    """Build ML data with `total_size` rows of which `complete_size` rows have
    a complete (non-NaN) target value.

    Target values equal input values on complete rows so that kNN predictions
    stay trivially bounded. Optional indices inject a missing input value on a
    target-missing row or a target-complete row.
    """
    if not 0 <= complete_size <= total_size:
        raise ValueError("complete_size must be in [0, total_size]")
    x = np.linspace(-2.0, 2.0, total_size)
    y = np.full(total_size, np.nan)
    y[:complete_size] = x[:complete_size]
    if missing_input_index is not None:
        x[missing_input_index] = np.nan
    if complete_input_index is not None:
        x[complete_input_index] = np.nan
    df = pd.DataFrame({"X": x, "Y": y})
    return DataFrameMLData(df, DatasetDescriptor("X", "Y"))


def make_kfold_ml_data(
    total_size: int = 120, complete_size: int = 60
) -> DataFrameMLData:
    """Build noisy data with enough complete rows for k-fold selection."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=total_size)
    y = x + rng.normal(scale=0.1, size=total_size)
    y[complete_size:] = np.nan
    return DataFrameMLData(
        pd.DataFrame({"X": x, "Y": y}),
        DatasetDescriptor("X", "Y"),
    )


class CountingKFold(SklearnKFold):
    """KFold recording construction and split calls."""

    split_calls = 0
    random_states: ClassVar[list[int | None]] = []

    def __init__(self, *args, **kwargs):
        type(self).random_states.append(kwargs.get("random_state"))
        super().__init__(*args, **kwargs)

    def split(self, X, y=None, groups=None):
        type(self).split_calls += 1
        return super().split(X, y, groups)


class RecordingScaler:
    """Scaler double recording fit and transform inputs per instance."""

    instances: ClassVar[list["RecordingScaler"]] = []
    kind: ClassVar[str] = "base"

    def __init__(self) -> None:
        self.fit_values: set[Any] | None = None
        self.transform_values: list[set[Any]] = []
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "RecordingScaler":
        self.fit_values = set(np.asarray(X).ravel())
        return self

    def transform(self, X: Any) -> np.ndarray:
        self.transform_values.append(set(np.asarray(X).ravel()))
        return np.asarray(X)

    def fit_transform(self, X: Any, y: Any = None, **kwargs: Any) -> np.ndarray:
        self.fit(X, y, **kwargs)
        return self.transform(X)


class RecordingStandardScaler(RecordingScaler):
    kind = "standardization"


class RecordingMinMaxScaler(RecordingScaler):
    kind = "normalization"


class TestOptimalKSelection(unittest.TestCase):
    """Regression tests for k selection on datasets with few complete rows.

    The k search must only count complete-case training rows, not the total
    dataset size, otherwise scikit-learn raises when the candidate k exceeds
    the number of training samples.
    """

    def test_kfold_with_single_complete_row(self):
        """10 rows, 1 complete: kfold selection must not crash and yield k=1."""
        sampler = KnnSampler(
            make_ml_data(total_size=10, complete_size=1),
            compute_bounds=False,
            optimal_k_method="kfold",
            strategy="mean",
        )
        sampler.fit()
        self.assertEqual(sampler.optimal_k, 1)

    def test_loo_with_few_complete_rows(self):
        """100 rows, 3 complete: candidate bound sqrt(100)=10 used to overflow
        the 3 training rows and crash the KNN fit; now capped to 1."""
        sampler = KnnSampler(
            make_ml_data(total_size=100, complete_size=3),
            compute_bounds=False,
            strategy="mean",
        )
        sampler.fit()
        self.assertEqual(sampler.optimal_k, 1)

    def test_kfold_falls_back_when_folds_exceed_complete_rows(self):
        """4 complete rows with 5 CV folds: falls back to penalized LOO."""
        sampler = KnnSampler(
            make_ml_data(total_size=10, complete_size=4),
            compute_bounds=False,
            optimal_k_method="kfold",
            optimal_k_cv_folds=5,
            strategy="mean",
        )
        sampler.fit()
        self.assertIn(sampler.optimal_k, {1, 2})

    def test_kfold_small_valid_case(self):
        """4 complete rows with 3 CV folds: regular kfold path stays in bounds."""
        sampler = KnnSampler(
            make_ml_data(total_size=10, complete_size=4),
            compute_bounds=False,
            optimal_k_method="kfold",
            optimal_k_cv_folds=3,
            strategy="mean",
        )
        sampler.fit()
        self.assertIn(sampler.optimal_k, {1, 2})

    def test_fit_with_empty_training_data_raises(self):
        """A dataset with no complete target row must raise a clear
        ValueError, not an obscure scikit-learn error."""
        sampler = KnnSampler(
            make_ml_data(total_size=10, complete_size=0),
            compute_bounds=False,
        )
        with self.assertRaisesRegex(ValueError, "complete-case training rows"):
            sampler.fit()


class TestExecutionReproducibility(unittest.TestCase):
    def make_sampler(self, data: DataFrameMLData | None = None) -> KnnSampler:
        return KnnSampler(
            data or make_ml_data(total_size=60, complete_size=40),
            compute_bounds=False,
            n_neighbors=3,
            strategy="sample",
        )

    def test_same_seed_reproduces_imputation(self):
        sampler = self.make_sampler()
        sampler.fit()

        first = sampler.execute(7)
        second = sampler.execute(7)

        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_differ(self):
        sampler = self.make_sampler()
        sampler.fit()

        first = sampler.execute(7)
        second = sampler.execute(8)

        self.assertFalse(first.equals(second))

    def test_numpy_integer_seed_is_honored(self):
        sampler = self.make_sampler()
        sampler.fit()

        first = sampler.execute(42)
        second = sampler.execute(cast(int, np.int64(42)))

        pd.testing.assert_frame_equal(first, second)

    def test_invalid_execution_seed_raises(self):
        sampler = self.make_sampler()
        sampler.fit()

        invalid_values: tuple[Any, ...] = (-1, 1.5, "42", True, np.bool_(True))
        for invalid in invalid_values:
            with self.subTest(random_state=invalid):
                with self.assertRaisesRegex(ValueError, "random_state"):
                    sampler.execute(invalid)

    def test_invalid_seed_raises_even_when_strategy_ignores_it(self):
        sampler = KnnSampler(
            make_ml_data(total_size=60, complete_size=40),
            compute_bounds=False,
            n_neighbors=3,
            strategy="mean",
        )
        sampler.fit()

        with self.assertRaisesRegex(ValueError, "random_state"):
            sampler.execute(cast(Any, 1.5))

    def test_invalid_seed_raises_without_missing_targets(self):
        sampler = KnnSampler(
            make_ml_data(total_size=20, complete_size=20),
            compute_bounds=False,
            n_neighbors=3,
            strategy="sample",
        )
        sampler.fit()

        with self.assertRaisesRegex(ValueError, "random_state"):
            sampler.execute(cast(Any, 1.5))


class TestInputValidation(unittest.TestCase):
    def test_missing_input_column_raises_on_fit(self):
        data: DataFrameMLData = DataFrameMLData(
            pd.DataFrame({"Y": [1.0, 2.0]}),
            DatasetDescriptor("X", "Y"),
        )
        sampler = KnnSampler(data, compute_bounds=False)

        with self.assertRaisesRegex(ValueError, "input column 'X'"):
            sampler.fit()

    def test_missing_target_column_raises_on_fit(self):
        data: DataFrameMLData = DataFrameMLData(
            pd.DataFrame({"X": [1.0, 2.0]}),
            DatasetDescriptor("X", "Y"),
        )
        sampler = KnnSampler(data, compute_bounds=False)

        with self.assertRaisesRegex(ValueError, "target column 'Y'"):
            sampler.fit()

    def test_missing_input_column_raises_on_impute_for_dataset(self):
        sampler = KnnSampler(
            make_ml_data(total_size=20, complete_size=10),
            compute_bounds=False,
            n_neighbors=3,
        )
        sampler.fit()

        with self.assertRaisesRegex(ValueError, "input column 'X'"):
            sampler.impute_for_dataset(
                pd.DataFrame({"Y": [np.nan]}),
                sampler.ml_data.nona_sets().y,
            )

    def test_missing_target_column_raises_on_impute_for_dataset(self):
        sampler = KnnSampler(
            make_ml_data(total_size=20, complete_size=10),
            compute_bounds=False,
            n_neighbors=3,
        )
        sampler.fit()

        with self.assertRaisesRegex(ValueError, r"Missing required column\(s\): 'Y'"):
            sampler.impute_for_dataset(
                pd.DataFrame({"X": [1.0]}),
                sampler.ml_data.nona_sets().y,
            )

    def test_missing_column_raises_on_execute_after_fit(self):
        sampler = KnnSampler(
            make_ml_data(total_size=20, complete_size=10),
            compute_bounds=False,
            n_neighbors=3,
        )
        sampler.fit()
        del sampler.ml_data.dataframe["X"]

        with self.assertRaisesRegex(ValueError, r"Missing required column\(s\): 'X'"):
            sampler.execute()

    def test_no_missing_targets_does_not_require_input_column(self):
        sampler = KnnSampler(
            make_ml_data(total_size=20, complete_size=10),
            compute_bounds=False,
            n_neighbors=3,
        )
        sampler.fit()

        imputed, bounds = sampler.impute_for_dataset(
            pd.DataFrame({"Y": [1.0, 2.0]}),
            sampler.ml_data.nona_sets().y,
        )

        self.assertTrue(imputed.empty)
        self.assertEqual(bounds[(5.0, 95.0)], ([], []))

    def test_missing_input_on_imputed_row_raises(self):
        sampler = KnnSampler(
            make_ml_data(total_size=60, complete_size=40, missing_input_index=45),
            compute_bounds=False,
            n_neighbors=3,
        )
        sampler.fit()

        with self.assertRaisesRegex(
            ValueError, "input column 'X' has missing values in 1 row"
        ):
            sampler.execute()

    def test_missing_input_on_training_row_is_ignored(self):
        sampler = KnnSampler(
            make_ml_data(total_size=60, complete_size=40, complete_input_index=0),
            compute_bounds=False,
            n_neighbors=3,
        )
        sampler.fit()

        result = sampler.execute(7)

        self.assertFalse(result["Y"].isna().any())


class TestKfoldScaling(unittest.TestCase):
    def test_scaler_is_fitted_on_each_training_fold(self):
        data = make_kfold_ml_data()
        train_sets = data.nona_sets()
        splits = list(
            SklearnKFold(n_splits=3, shuffle=True, random_state=7).split(train_sets.x)
        )
        expected_fit_values = [
            set(np.asarray(train_sets.x.iloc[train_idx]).ravel())
            for train_idx, _test_idx in splits
        ]
        expected_transform_values = [
            set(np.asarray(train_sets.x.iloc[test_idx]).ravel())
            for _train_idx, test_idx in splits
        ]

        for scaling, expected_kind, expected_type, other_type in (
            (
                "standardization",
                "standardization",
                RecordingStandardScaler,
                RecordingMinMaxScaler,
            ),
            (
                "normalization",
                "normalization",
                RecordingMinMaxScaler,
                RecordingStandardScaler,
            ),
        ):
            with self.subTest(scaling=scaling):
                RecordingStandardScaler.reset()
                RecordingMinMaxScaler.reset()
                sampler = KnnSampler(
                    data,
                    compute_bounds=False,
                    strategy="mean",
                    optimal_k_method="kfold",
                    optimal_k_random_state=7,
                    scaling_optimal_k=cast(Any, scaling),
                    scaling_fit="none",
                )

                with (
                    patch(
                        "src.imputation.knnsampler.StandardScaler",
                        RecordingStandardScaler,
                    ),
                    patch(
                        "src.imputation.knnsampler.MinMaxScaler",
                        RecordingMinMaxScaler,
                    ),
                ):
                    sampler.find_optimal_k_kfold(train_sets)

                self.assertEqual(other_type.instances, [])
                self.assertEqual(len(expected_type.instances), len(splits))
                for scaler, fit_values, transform_values in zip(
                    expected_type.instances,
                    expected_fit_values,
                    expected_transform_values,
                    strict=True,
                ):
                    self.assertEqual(scaler.kind, expected_kind)
                    self.assertEqual(scaler.fit_values, fit_values)
                    self.assertEqual(
                        scaler.transform_values,
                        [fit_values, transform_values],
                    )

    def test_none_scaling_does_not_instantiate_scaler(self):
        data = make_kfold_ml_data()
        train_sets = data.nona_sets()
        sampler = KnnSampler(
            data,
            compute_bounds=False,
            strategy="mean",
            optimal_k_method="kfold",
            optimal_k_random_state=7,
            scaling_optimal_k="none",
            scaling_fit="none",
        )

        with (
            patch(
                "src.imputation.knnsampler.StandardScaler",
                side_effect=AssertionError("StandardScaler constructed"),
            ),
            patch(
                "src.imputation.knnsampler.MinMaxScaler",
                side_effect=AssertionError("MinMaxScaler constructed"),
            ),
        ):
            selected = sampler.find_optimal_k_kfold(train_sets)

        self.assertGreaterEqual(selected, 1)


class TestKfoldSelection(unittest.TestCase):
    def test_default_seed_is_deterministic(self):
        selected = []
        for _ in range(3):
            sampler = KnnSampler(
                make_kfold_ml_data(),
                compute_bounds=False,
                strategy="mean",
                optimal_k_method="kfold",
            )
            self.assertEqual(sampler.optimal_k_random_state, DEFAULT_RANDOM_STATE)
            sampler.fit()
            selected.append(sampler.optimal_k)

        self.assertEqual(len(set(selected)), 1)

    def test_loo_penalized_ignores_optimal_k_random_state(self):
        selected = []
        for seed in (None, 0, 7, MAX_KFOLD_RANDOM_STATE):
            sampler = KnnSampler(
                make_kfold_ml_data(),
                compute_bounds=False,
                strategy="mean",
                optimal_k_method="loo_penalized",
                optimal_k_random_state=seed,
            )
            sampler.fit()
            selected.append(sampler.optimal_k)

        self.assertEqual(len(set(selected)), 1)

    def test_explicit_seed_is_forwarded_and_folds_are_materialized_once(self):
        CountingKFold.split_calls = 0
        CountingKFold.random_states = []
        sampler = KnnSampler(
            make_kfold_ml_data(),
            compute_bounds=False,
            strategy="mean",
            optimal_k_method="kfold",
            optimal_k_random_state=7,
        )

        with patch("src.imputation.knnsampler.KFold", CountingKFold):
            sampler.fit()

        self.assertEqual(CountingKFold.random_states, [7])
        self.assertEqual(CountingKFold.split_calls, 1)

    def test_none_seed_is_forwarded_and_folds_are_materialized_once(self):
        CountingKFold.split_calls = 0
        CountingKFold.random_states = []
        sampler = KnnSampler(
            make_kfold_ml_data(),
            compute_bounds=False,
            strategy="mean",
            optimal_k_method="kfold",
            optimal_k_random_state=None,
        )

        with patch("src.imputation.knnsampler.KFold", CountingKFold):
            sampler.fit()

        self.assertEqual(CountingKFold.random_states, [None])
        self.assertEqual(CountingKFold.split_calls, 1)

    def test_invalid_random_state_raises(self):
        invalid_values: tuple[Any, ...] = (-1, 1.5, "42", True, np.bool_(True))
        for invalid in invalid_values:
            with self.subTest(random_state=invalid):
                with self.assertRaisesRegex(ValueError, "optimal_k_random_state"):
                    KnnSampler(
                        make_kfold_ml_data(),
                        compute_bounds=False,
                        optimal_k_method="kfold",
                        optimal_k_random_state=invalid,
                    )

    def test_numpy_integer_random_state_is_accepted(self):
        sampler = KnnSampler(
            make_kfold_ml_data(),
            compute_bounds=False,
            optimal_k_method="kfold",
            optimal_k_random_state=cast(int, np.int64(7)),
        )

        self.assertEqual(sampler.optimal_k_random_state, 7)
        self.assertIsInstance(sampler.optimal_k_random_state, int)

    def test_random_state_upper_bound(self):
        sampler = KnnSampler(
            make_kfold_ml_data(),
            compute_bounds=False,
            optimal_k_method="kfold",
            optimal_k_random_state=MAX_KFOLD_RANDOM_STATE,
        )
        self.assertEqual(sampler.optimal_k_random_state, MAX_KFOLD_RANDOM_STATE)

        with self.assertRaisesRegex(ValueError, r"<= 2\*\*32 - 1"):
            KnnSampler(
                make_kfold_ml_data(),
                compute_bounds=False,
                optimal_k_method="kfold",
                optimal_k_random_state=MAX_KFOLD_RANDOM_STATE + 1,
            )

    def test_random_state_upper_bound_is_kfold_only(self):
        seed = MAX_KFOLD_RANDOM_STATE + 1
        sampler = KnnSampler(
            make_kfold_ml_data(),
            compute_bounds=False,
            optimal_k_method="loo_penalized",
            optimal_k_random_state=seed,
        )

        self.assertEqual(sampler.optimal_k_random_state, seed)

    def test_kfold_seed_bound_is_rechecked_at_fit(self):
        sampler = KnnSampler(
            make_kfold_ml_data(),
            compute_bounds=False,
            strategy="mean",
            optimal_k_method="loo_penalized",
            optimal_k_random_state=MAX_KFOLD_RANDOM_STATE + 1,
        )
        sampler.optimal_k_method = "kfold"

        with self.assertRaisesRegex(ValueError, r"<= 2\*\*32 - 1"):
            sampler.fit()

    def test_kfold_seed_type_is_rechecked_at_fit(self):
        for invalid in (-5, 1.5, "42", True, np.bool_(True)):
            with self.subTest(random_state=invalid):
                sampler = KnnSampler(
                    make_kfold_ml_data(),
                    compute_bounds=False,
                    strategy="mean",
                    optimal_k_method="loo_penalized",
                    optimal_k_random_state=7,
                )
                sampler.optimal_k_random_state = cast(Any, invalid)
                sampler.optimal_k_method = "kfold"

                with self.assertRaisesRegex(ValueError, "optimal_k_random_state"):
                    sampler.fit()


if __name__ == "__main__":
    unittest.main()
