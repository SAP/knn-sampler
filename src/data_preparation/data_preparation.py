from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, Union

import numpy as np
import pandas as pd

from src.data_preparation.data_description import (
    DataFrameMLData,
)
from src.data_preparation.dataset_description import (
    DatasetDescriptor,
    ExcelDatasetDescriptor,
)

MLT = TypeVar("MLT")
DT = TypeVar("DT", bound=DatasetDescriptor)


class Rate:
    def __init__(self, nb: float) -> None:
        if not 0 <= nb <= 1:
            raise Exception("Rate should be between 0 and 1")
        self._nb = nb

    @property
    def nb(self):
        return self._nb

    @nb.setter
    def nb(self, nb: float):
        if not 0 <= nb <= 1:
            raise Exception("Rate should be between 0 and 1")
        self._nb = nb

    def __str__(self) -> str:
        return f"rate of {self.nb}"


class MissingDataGenerator(ABC):
    @abstractmethod
    def create_missing_data(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        pass


@dataclass
class NoAdditionalMissingValues(MissingDataGenerator):
    def __str__(self) -> str:
        return "No missing values"

    def __repr__(self) -> str:
        return self.__str__()

    def create_missing_data(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        return np.array([]), pd.Series()


@dataclass
class Mcar(MissingDataGenerator):
    def __init__(self, missing_values: Union[int, Rate]) -> None:
        super().__init__()
        self.missing_values = missing_values

    def create_missing_data(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        size = (
            int(len(df.index) * self.missing_values.nb)
            if isinstance(self.missing_values, Rate)
            else self.missing_values
        )
        missing_indexes = np.random.choice(
            df.index,
            size=size,
            replace=False,
        )
        actual_values = df.loc[missing_indexes, "Y"].copy()
        df.loc[missing_indexes, "Y"] = np.nan
        return missing_indexes, actual_values

    def __str__(self) -> str:
        return "MCAR"

    def __repr__(self) -> str:
        return "MCAR"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Mcar)

    def __hash__(self) -> int:
        return hash(Mcar)


@dataclass
class Mar(MissingDataGenerator):
    def __init__(
        self, chunk_start: float, chunk_end: float, missing_values: Union[int, Rate]
    ) -> None:
        super().__init__()
        self.chunk_start = chunk_start
        self.chunk_end = chunk_end
        self.missing_values = missing_values

    def create_missing_data(self, df: pd.DataFrame):
        chunk_indexes = df[
            (df["X"] > self.chunk_start) & (df["X"] < self.chunk_end)
        ].index
        chunk_size = len(chunk_indexes)
        if chunk_size == 0:
            return np.array([]), pd.Series()
        size = (
            int(chunk_size * self.missing_values.nb)
            if isinstance(self.missing_values, Rate)
            else min(chunk_size, self.missing_values)
        )
        if size == 0:
            return np.array([]), pd.Series()
        missing_indexes = np.random.choice(
            chunk_indexes,
            size=size,
            replace=False,
        )
        actual_values = df.loc[missing_indexes, "Y"].copy()
        df.loc[missing_indexes, "Y"] = np.nan
        return missing_indexes, actual_values

    def __str__(self) -> str:
        return f"MAR : chunk start = {self.chunk_start}, chunk end = {self.chunk_end}"

    def __repr__(self) -> str:
        return f"MAR : chunk start = {self.chunk_start}, chunk end = {self.chunk_end}"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Mar) and (value.chunk_start, value.chunk_end) == (
            self.chunk_start,
            self.chunk_end,
        )

    def __hash__(self) -> int:
        return hash((self.chunk_start, self.chunk_end))


class DataPreparator(ABC, Generic[MLT, DT]):
    def __init__(
        self,
        dataset_descriptor: DT,
        missing_generator: MissingDataGenerator,
        sample_size: Union[int, None],
    ):
        self.dataset_descriptor = dataset_descriptor
        self.missing_type = missing_generator
        self.sample_size = sample_size

        self.missing_indexes: np.ndarray
        self.actual_values: pd.Series

    @abstractmethod
    def prepare_data(self) -> MLT:
        raise NotImplementedError

    def __str__(self) -> str:
        return (
            f"{self.dataset_descriptor} \n"
            f"\tmissing generator: {self.missing_type} \n"
            f"\tsample size: {self.sample_size}"
        )


class ExcelDataPreparator(DataPreparator[DataFrameMLData, ExcelDatasetDescriptor]):
    def __init__(
        self,
        dataset_descriptor: ExcelDatasetDescriptor,
        missing_generator: MissingDataGenerator,
        sample_size: Union[int, None] = None,
        columns_to_keep: Union[list[str], None] = None,
    ):
        super().__init__(
            dataset_descriptor=dataset_descriptor,
            missing_generator=missing_generator,
            sample_size=sample_size,
        )
        self.columns_to_keep = columns_to_keep if columns_to_keep is not None else []

    def prepare_data(self) -> DataFrameMLData:
        if self.sample_size is None:
            df = pd.read_excel(
                io=self.dataset_descriptor.path,
                usecols=[
                    self.dataset_descriptor.input_column,
                    self.dataset_descriptor.target_column,
                    *self.columns_to_keep,
                ],
            )
            self.sample_size = len(df)
        else:
            df = pd.read_excel(
                io=self.dataset_descriptor.path,
                usecols=[
                    self.dataset_descriptor.input_column,
                    self.dataset_descriptor.target_column,
                    *self.columns_to_keep,
                ],
            ).sample(self.sample_size)

        # rename columns
        df = df.rename(
            columns={
                self.dataset_descriptor.input_column: "X",
                self.dataset_descriptor.target_column: "Y",
            }
        )

        # sort by index and reset it
        df = df.sort_values(by="X").reset_index(drop=True)

        self.missing_indexes, self.actual_values = (
            self.missing_type.create_missing_data(df)
        )

        return DataFrameMLData(
            df, DatasetDescriptor("X", "Y", other_columns=self.columns_to_keep)
        )


class DataGenerator(DataPreparator[DataFrameMLData, DatasetDescriptor]):
    def __init__(
        self,
        linear_interpolation_ratio: float,
        sample_size: int,
        missing_generator: MissingDataGenerator,
        geometry_type: Literal["linear"] | Literal["ring"] = "linear",
        dataset_descriptor=None,
    ):
        super().__init__(
            dataset_descriptor=dataset_descriptor or DatasetDescriptor("X", "Y"),
            missing_generator=missing_generator,
            sample_size=sample_size,
        )
        self.linear_interpolation_ratio = linear_interpolation_ratio
        self.geometry_type: Literal["linear"] | Literal["ring"] = geometry_type
        self.sample_size: int

    def build_synthetic_data(self) -> pd.DataFrame:
        match self.geometry_type:
            case "linear":
                x = np.linspace(-2, 2, self.sample_size)
                epsilon = np.random.chisquare(2, self.sample_size)
                y = self.linear_interpolation_ratio * x + epsilon

            case "ring":
                theta = np.linspace(0, 2 * np.pi, self.sample_size)
                epsilon = np.random.normal(0, 0.1, self.sample_size)
                r = 1 + epsilon
                x = np.cos(theta) * r
                y = np.sin(theta) * r

        return pd.DataFrame(
            {
                self.dataset_descriptor.input_column: x,
                self.dataset_descriptor.target_column: y,
            }
        )

    def prepare_data(self) -> DataFrameMLData:
        df = self.build_synthetic_data()
        self.missing_indexes, self.actual_values = (
            self.missing_type.create_missing_data(df)
        )
        return DataFrameMLData(df, self.dataset_descriptor)
