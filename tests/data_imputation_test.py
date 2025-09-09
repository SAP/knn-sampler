import unittest
from typing import Literal

import pandas as pd

from main import do_plot_results
from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import (
    DataGenerator,
    ExcelDataPreparator,
    Rate,
)
from src.imputation import (
    Imputer,
    KnnImputer,
    KnnSampler,
    KNNxKDEImputer,
    LinearImputer,
    MICEImputer,
    RandomForestImputer,
)
from src.utils import do_imputation
from tests.testutils import prepare_data_from_dataset, prepare_generated_data


class TestImputation(unittest.TestCase):
    dataset_results: dict[Imputer, tuple[pd.DataFrame, float | None]]
    generated_results: dict[Imputer, tuple[pd.DataFrame, float | None]]
    prep_dataset: DataFrameMLData
    prep_gen: DataFrameMLData
    act_dataset: pd.Series
    act_gen: pd.Series
    preparator: ExcelDataPreparator
    missing_values: int | Rate
    generator: DataGenerator
    imputers: dict[Literal["dataset"] | Literal["generated"], list[Imputer]]

    @classmethod
    def setUpClass(cls):
        def setupImputers(prepared_data: DataFrameMLData) -> list[Imputer]:
            return [
                KnnSampler(
                    ml_data=prepared_data,
                    lower_percentiles=[5],
                    upper_percentiles=[95],
                    compute_bounds=True,
                ),
                KnnImputer(ml_data=prepared_data),
                LinearImputer(ml_data=prepared_data),
                MICEImputer(ml_data=prepared_data),
                RandomForestImputer(ml_data=prepared_data),
                KNNxKDEImputer(ml_data=prepared_data),
            ]

        cls.prep_dataset, cls.act_dataset, cls.preparator, cls.missing_values = (
            prepare_data_from_dataset()
        )
        cls.prep_gen, cls.act_gen, cls.generator = prepare_generated_data()
        cls.imputers = {
            "dataset": setupImputers(cls.prep_dataset),
            "generated": setupImputers(cls.prep_gen),
        }

        cls.dataset_results = {}
        cls.generated_results = {}

    def test_data_preparation(self):
        with self.subTest(source="dataset"):
            self.assertEqual(self.act_dataset.isnull().sum(), 0)
            self.assertGreater(self.prep_dataset.df.isnull().sum().sum(), 0)
        with self.subTest(source="generated"):
            self.assertEqual(self.act_gen.isnull().sum(), 0)
            self.assertGreater(self.prep_gen.df.isnull().sum().sum(), 0)

    def test_impute_with_dataset_data(self):
        for imputer in self.imputers["dataset"]:
            with self.subTest(source=imputer.__class__.__name__):
                imputer.fit()
                results = imputer.execute()
                self.dataset_results[imputer] = results, None
                self.assertIsNotNone(results)

    def test_impute_with_generated_data(self):
        for imputer in self.imputers["generated"]:
            with self.subTest(source=imputer.__class__.__name__):
                imputer.fit()
                results = imputer.execute()
                self.generated_results[imputer] = results, None
                self.assertIsNotNone(results)

    def test_plot_results(self):
        with self.subTest(source="plot dataset"):
            do_plot_results(
                results=self.dataset_results,
                missing_values=self.missing_values,
                data=self.prep_dataset,
                actual_values=self.act_dataset,
                block=False,
            )
        with self.subTest(source="plot generated"):
            do_plot_results(
                results=self.generated_results,
                missing_values=self.missing_values,
                data=self.prep_gen,
                actual_values=self.act_gen,
                block=False,
            )


class TestMultipleImputation(unittest.TestCase):
    prep_gen: DataFrameMLData
    act_gen: pd.Series
    generator: DataGenerator

    @classmethod
    def setUpClass(cls):
        cls.prep_gen, cls.act_gen, cls.generator = prepare_generated_data()

    def test_multiple_imputation_knn_sampler(self):
        nb_iterations = 5
        sampler = KnnSampler(
            ml_data=self.prep_gen,
            lower_percentiles=[5],
            upper_percentiles=[95],
            compute_bounds=True,
        )
        results = do_imputation(sampler, iterations=nb_iterations)
        # assert each element are of different sizes because knnsampler should produce different imputations based on random

        set_result = {
            (tuple(map(tuple, df.values)), tuple(map(tuple, df.columns)))
            for df, _ in results[sampler]
        }
        self.assertEqual(nb_iterations, len(set_result))

    def test_multiple_imputation_deterministic(self):
        nb_iterations = 5
        imputer = KnnImputer(ml_data=self.prep_gen)
        results = do_imputation(imputer, iterations=nb_iterations)
        # assert each element are of size 1 because KnnImputer should always produce the same data imputation
        set_result = {
            (tuple(map(tuple, df.values)), tuple(map(tuple, df.columns)))
            for df, _ in results[imputer]
        }
        self.assertEqual(1, len(set_result))
