import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.data_preparation.data_preparation import (
    DataGenerator,
    ExcelDataPreparator,
    GeometryType,
    Mar,
    Mcar,
    NoAdditionalMissingValues,
    Rate,
    split_random_state,
)
from src.data_preparation.dataset_description import (
    DatasetDescriptor,
    Datasets,
    ExcelDatasetDescriptor,
)


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


@pytest.mark.parametrize("sample_size", [None, 4, 8])
@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
def test_excel_replays_duplicate_x_with_requested_size(
    tmp_path, sample_size, mechanism
):
    source = tmp_path / "duplicates.xlsx"
    pd.DataFrame({"input": [1.0] * 8, "target": range(8)}).to_excel(source, index=False)
    preparation, missingness = split_random_state(42)

    def build():
        missing = (
            Mcar(2, random_state=missingness)
            if mechanism == "MCAR"
            else Mar(0, 2, 2, random_state=missingness)
        )
        return ExcelDataPreparator(
            ExcelDatasetDescriptor("input", "target", str(source), "Sheet1"),
            missing,
            sample_size=sample_size,
            random_state=preparation,
        )

    preparator = build()
    first = preparator.prepare_data().df
    actual = preparator.actual_values.copy()
    indexes = preparator.missing_indexes.copy()
    for current in (preparator, build()):
        repeated = current.prepare_data().df
        pd.testing.assert_frame_equal(first, repeated)
        pd.testing.assert_series_equal(actual, current.actual_values)
        np.testing.assert_array_equal(indexes, current.missing_indexes)
        pd.testing.assert_series_equal(first.Y.isna(), repeated.Y.isna())
        assert current.sample_size == sample_size


def test_excel_all_rows_never_samples_even_without_seed(tmp_path, monkeypatch):
    source = tmp_path / "all.xlsx"
    pd.DataFrame({"X": [1.0, 1.0], "Y": [2.0, 3.0]}).to_excel(source, index=False)
    preparator = ExcelDataPreparator(
        ExcelDatasetDescriptor("X", "Y", str(source), "Sheet1"),
        NoAdditionalMissingValues(),
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "sample",
        lambda *a, **k: pytest.fail("all rows must not be sampled"),
    )
    pd.testing.assert_frame_equal(
        preparator.prepare_data().df, preparator.prepare_data().df
    )
    assert preparator.sample_size is None


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

    def test_mcar_missing_values_overflow_rejected(self):
        """An absolute count larger than the dataset must raise a clear
        error instead of an opaque numpy one."""
        df = pd.DataFrame({"X": range(10), "Y": range(10)})
        with self.assertRaisesRegex(ValueError, "exceeds dataset size"):
            Mcar(500).create_missing_data(df)


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
