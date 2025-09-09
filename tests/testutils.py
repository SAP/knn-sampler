import pandas as pd

from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import (
    DataGenerator,
    ExcelDataPreparator,
    Mcar,
    Rate,
)
from src.data_preparation.dataset_description import Datasets


def prepare_data_from_dataset() -> tuple[
    DataFrameMLData, pd.Series, ExcelDataPreparator, int | Rate
]:
    return (
        (
            preparator := ExcelDataPreparator(
                dataset_descriptor=Datasets.PUBLICATION_DATASET,
                sample_size=1000,
                missing_generator=Mcar(missing_values := (Rate(0.3))),
            )
        ).prepare_data(),
        preparator.actual_values,
        preparator,
        missing_values,
    )


def prepare_generated_data() -> tuple[DataFrameMLData, pd.Series, DataGenerator]:
    return (
        (generator := DataGenerator(1.0, 1000, Mcar(Rate(0.3)))).prepare_data(),
        generator.actual_values,
        generator,
    )
