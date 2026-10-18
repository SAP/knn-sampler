import unittest
from typing import cast

import pandas as pd

from src.data_preparation.data_preparation import (
    DataGenerator,
    ExcelDataPreparator,
    GeometryType,
    Mar,
    Mcar,
    Rate,
)
from src.data_preparation.dataset_description import Datasets


class TestDataPreparation(unittest.TestCase):
    def test_data_generation_MCAR(self):
        generator = DataGenerator(
            linear_interpolation_ratio=1,
            sample_size=1000,
            missing_generator=Mcar(Rate(0.3)),
        )
        data = generator.prepare_data()
        self.assertIsNotNone(data)
        self.assertIsNotNone(generator.actual_values)

    def test_excel_data_preparation_MCAR(self):
        preparator = ExcelDataPreparator(
            dataset_descriptor=Datasets.PUBLICATION_DATASET,
            sample_size=1000,
            missing_generator=Mcar(Rate(0.3)),
        )
        data = preparator.prepare_data()
        self.assertIsNotNone(data)
        self.assertIsNotNone(preparator.actual_values)

    def test_data_generation_MAR(self):
        generator = DataGenerator(
            linear_interpolation_ratio=1,
            sample_size=1000,
            missing_generator=Mar(0.5, 1.5, Rate(0.3)),
        )
        data = generator.prepare_data()
        self.assertIsNotNone(data)
        self.assertIsNotNone(generator.actual_values)

    def test_excel_data_preparation_MAR(self):
        preparator = ExcelDataPreparator(
            dataset_descriptor=Datasets.PUBLICATION_DATASET,
            sample_size=1000,
            missing_generator=Mar(0.5, 1.5, Rate(0.3)),
        )
        data = preparator.prepare_data()
        self.assertIsNotNone(data)
        self.assertIsNotNone(preparator.actual_values)

    def test_data_generation_MAR_with_absolute_missing_values(self):
        """Test that MAR works with absolute number of missing values (int) like MCAR does"""
        generator = DataGenerator(
            linear_interpolation_ratio=1,
            sample_size=1000,
            missing_generator=Mar(0.5, 1.5, missing_values=100),
        )
        data = generator.prepare_data()
        self.assertIsNotNone(data)
        self.assertIsNotNone(generator.actual_values)
        # Should have at most 100 missing values (could be less if chunk is smaller)
        self.assertLessEqual(len(generator.actual_values), 100)


class TestRateValidationAndEquality(unittest.TestCase):
    def test_rate_validation(self):
        with self.assertRaises(ValueError):
            Rate(1.5)
        with self.assertRaises(ValueError):
            Rate(-0.1)
        with self.assertRaises(ValueError):
            rate = Rate(0.3)
            rate.nb = 2.0

    def test_rate_equality(self):
        self.assertEqual(Rate(0.3), Rate(0.3))
        self.assertNotEqual(Rate(0.3), Rate(0.5))
        # A Rate never equals a raw float
        self.assertNotEqual(Rate(0.3), 0.3)
        self.assertEqual(hash(Rate(0.3)), hash(Rate(0.3)))

    def test_mcar_equality(self):
        self.assertEqual(Mcar(Rate(0.3)), Mcar(Rate(0.3)))
        self.assertEqual(Mcar(100), Mcar(100))
        self.assertNotEqual(Mcar(Rate(0.3)), Mcar(Rate(0.5)))
        # Mixed int vs Rate missing values compare unequal
        self.assertNotEqual(Mcar(Rate(0.3)), Mcar(100))
        self.assertNotEqual(Mcar(Rate(0.3)), Mar(0.5, 1.5, Rate(0.3)))
        self.assertEqual(hash(Mcar(Rate(0.3))), hash(Mcar(Rate(0.3))))

    def test_mar_equality(self):
        self.assertEqual(Mar(0.5, 1.5, Rate(0.3)), Mar(0.5, 1.5, Rate(0.3)))
        self.assertNotEqual(Mar(0.5, 1.5, Rate(0.3)), Mar(0.4, 1.5, Rate(0.3)))
        self.assertNotEqual(Mar(0.5, 1.5, Rate(0.3)), Mar(0.5, 1.5, Rate(0.5)))
        self.assertNotEqual(Mar(0.5, 1.5, Rate(0.3)), Mcar(Rate(0.3)))
        self.assertEqual(hash(Mar(0.5, 1.5, Rate(0.3))), hash(Mar(0.5, 1.5, Rate(0.3))))

    def test_negative_missing_values_rejected(self):
        """Negative counts used to fail later with an opaque numpy error."""
        with self.assertRaises(ValueError):
            Mcar(-5)
        with self.assertRaises(ValueError):
            Mar(0.5, 1.5, -5)


class TestSeededReproducibility(unittest.TestCase):
    def test_data_generator_reproducible_with_seed(self):
        """Two freshly built seeded generators produce identical data."""

        def build():
            generator = DataGenerator(
                linear_interpolation_ratio=1,
                sample_size=200,
                missing_generator=Mcar(Rate(0.3), random_state=42),
                random_state=42,
            )
            return generator, generator.prepare_data()

        generator1, data1 = build()
        generator2, data2 = build()
        pd.testing.assert_frame_equal(data1.df, data2.df)
        self.assertEqual(
            list(generator1.missing_indexes), list(generator2.missing_indexes)
        )

    def test_data_generator_repeated_prepare_is_deterministic(self):
        """prepare_data is idempotent under a fixed seed (per-call seeding)."""
        generator = DataGenerator(
            linear_interpolation_ratio=1,
            sample_size=200,
            missing_generator=Mcar(Rate(0.3), random_state=42),
            random_state=42,
        )
        first = generator.prepare_data()
        first_missing = list(generator.missing_indexes)
        second = generator.prepare_data()
        self.assertEqual(first_missing, list(generator.missing_indexes))
        pd.testing.assert_frame_equal(first.df, second.df)

    def test_mar_reproducible_with_seed(self):
        def build():
            generator = DataGenerator(
                linear_interpolation_ratio=1,
                sample_size=200,
                missing_generator=Mar(0.5, 1.5, Rate(0.3), random_state=42),
                random_state=42,
            )
            return generator.prepare_data(), generator.actual_values

        (_data1, actual1), (_data2, actual2) = build(), build()
        pd.testing.assert_series_equal(actual1, actual2)

    def test_excel_preparation_reproducible_with_seed(self):
        def build():
            preparator = ExcelDataPreparator(
                dataset_descriptor=Datasets.PUBLICATION_DATASET,
                sample_size=1000,
                missing_generator=Mcar(Rate(0.3), random_state=42),
                random_state=42,
            )
            return preparator.prepare_data(), preparator.actual_values

        (data1, actual1), (data2, actual2) = build(), build()
        pd.testing.assert_frame_equal(data1.df, data2.df)
        pd.testing.assert_series_equal(actual1, actual2)


class TestGeometryValidation(unittest.TestCase):
    def test_invalid_geometry_type(self):
        """Invalid geometry used to crash with UnboundLocalError at build time."""
        with self.assertRaises(ValueError):
            DataGenerator(
                linear_interpolation_ratio=1,
                sample_size=100,
                missing_generator=Mcar(Rate(0.3)),
                geometry_type=cast(GeometryType, "invalid"),
            )
