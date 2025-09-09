from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd

from src.data_preparation.dataset_description import (
    DatasetDescriptor,
)


@dataclass
class MLSets:
    x: pd.DataFrame
    y: pd.Series


# Dataset descriptor generic type
DDT = TypeVar("DDT", bound=DatasetDescriptor)


class DataFrameMLData(Generic[DDT]):
    def __init__(self, df: pd.DataFrame, dataset_descriptor: DatasetDescriptor):
        self.dataframe = df
        self.dataset_descriptor: DatasetDescriptor = dataset_descriptor

    @property
    def df(self) -> pd.DataFrame:
        return self.dataframe[
            [
                self.dataset_descriptor.input_column,
                self.dataset_descriptor.target_column,
                *self.dataset_descriptor.other_columns,
            ]
        ]

    @df.setter
    def df(self, df: pd.DataFrame):
        self.dataframe = df

    def get_ml_sets(self) -> MLSets:
        x = self.dataframe[[self.dataset_descriptor.input_column]]
        y = self.dataframe[self.dataset_descriptor.target_column]
        return MLSets(x, y)

    def nona_sets(self) -> MLSets:
        filtered_df = self.dataframe.dropna()
        x = filtered_df[[self.dataset_descriptor.input_column]]
        y = filtered_df[self.dataset_descriptor.target_column]
        return MLSets(x, y)
