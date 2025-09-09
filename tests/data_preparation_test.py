import unittest

from src.data_preparation.data_preparation import (
    DataGenerator,
    ExcelDataPreparator,
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
