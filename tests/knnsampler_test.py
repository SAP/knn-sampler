import unittest

import numpy as np
import pandas as pd

from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.dataset_description import DatasetDescriptor
from src.imputation.knnsampler import KnnSampler


def make_ml_data(total_size: int, complete_size: int) -> DataFrameMLData:
    """Build ML data with `total_size` rows of which `complete_size` rows have
    a complete (non-NaN) target value.

    Target values equal input values on complete rows so that kNN predictions
    stay trivially bounded.
    """
    if not 0 <= complete_size <= total_size:
        raise ValueError("complete_size must be in [0, total_size]")
    x = np.linspace(-2.0, 2.0, total_size)
    y = np.full(total_size, np.nan)
    y[:complete_size] = x[:complete_size]
    df = pd.DataFrame({"X": x, "Y": y})
    return DataFrameMLData(df, DatasetDescriptor("X", "Y"))


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


if __name__ == "__main__":
    unittest.main()
