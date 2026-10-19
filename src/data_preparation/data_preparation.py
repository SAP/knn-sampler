from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.data_preparation.data_description import (
    DataFrameMLData,
)
from src.data_preparation.dataset_description import (
    DatasetDescriptor,
    ExcelDatasetDescriptor,
)

GeometryType = Literal["linear", "ring"]
GEOMETRY_TYPES: tuple[GeometryType, ...] = ("linear", "ring")


def split_random_state(random_state: int | None) -> tuple[int | None, int | None]:
    """Derive replayable preparation/missingness seeds from separate child streams.

    Child seeds fit uint32 for pandas sampling. None keeps both steps unseeded.
    Pass the returned seeds explicitly to preparation and missingness generators;
    np.random.seed() does not control synthetic generation or MCAR/MAR selection.
    """
    if random_state is None:
        return None, None
    preparation, missingness = np.random.SeedSequence(random_state).spawn(2)
    return int(preparation.generate_state(1)[0]), int(missingness.generate_state(1)[0])


class Rate:
    """Missing-value fraction in [0, 1]. Do not mutate while used as a dict/set key."""

    def __init__(self, nb: float) -> None:
        if not 0 <= nb <= 1:
            raise ValueError("Rate should be between 0 and 1")
        self._nb = nb

    @property
    def nb(self) -> float:
        return self._nb

    @nb.setter
    def nb(self, nb: float) -> None:
        if not 0 <= nb <= 1:
            raise ValueError("Rate should be between 0 and 1")
        self._nb = nb

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Rate) and self._nb == value._nb

    def __hash__(self) -> int:
        # Value equality is mutable: the hash must not depend on nb.
        return hash(Rate)

    def __str__(self) -> str:
        return f"rate of {self.nb}"


class MissingDataGenerator(ABC):
    """Inject missing values; do not mutate instances while used as dict/set keys."""

    @abstractmethod
    def create_missing_data(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        raise NotImplementedError


@dataclass
class NoAdditionalMissingValues(MissingDataGenerator):
    def __str__(self) -> str:
        return "No missing values"

    def __repr__(self) -> str:
        return self.__str__()

    def create_missing_data(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        return np.array([]), pd.Series()


class Mcar(MissingDataGenerator):
    """Missing Completely At Random missing-data generator.

    Parameters
    ----------
    missing_values : int | Rate
        Absolute number of missing values or missing rate in [0, 1].
    random_state : int | None
        Seed for reproducible missing-value selection. Each call to
        create_missing_data replays the stream seeded with it;
        None yields nondeterministic draws.
    """

    def __init__(
        self,
        missing_values: int | Rate,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(missing_values, int) and missing_values < 0:
            raise ValueError(
                f"missing_values should be a non-negative integer or a Rate,"
                f" got '{missing_values}'"
            )
        self.missing_values = missing_values
        self.random_state = random_state

    def create_missing_data(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        size = (
            int(len(df.index) * self.missing_values.nb)
            if isinstance(self.missing_values, Rate)
            else self.missing_values
        )
        if size > len(df.index):
            raise ValueError(
                f"missing_values ({size}) exceeds dataset size ({len(df.index)})"
            )
        rng = np.random.default_rng(self.random_state)
        missing_indexes = rng.choice(df.index.to_numpy(), size=size, replace=False)
        actual_values = df.loc[missing_indexes, "Y"].copy()
        df.loc[missing_indexes, "Y"] = np.nan
        return missing_indexes, actual_values

    def __str__(self) -> str:
        return "MCAR"

    def __repr__(self) -> str:
        return "MCAR"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Mcar) and (
            self.missing_values == value.missing_values
            and self.random_state == value.random_state
        )

    def __hash__(self) -> int:
        return hash(Mcar)


class Mar(MissingDataGenerator):
    """Missing At Random missing-data generator.

    Missing values are drawn from the index range where X lies within
    (chunk_start, chunk_end).

    Parameters
    ----------
    chunk_start : float
        Lower bound of the X range affected by missing values.
    chunk_end : float
        Upper bound of the X range affected by missing values.
    missing_values : int | Rate
        Absolute number of missing values or missing rate in [0, 1].
    random_state : int | None
        Seed for reproducible missing-value selection. Each call to
        create_missing_data replays the stream seeded with it;
        None yields nondeterministic draws.
    """

    def __init__(
        self,
        chunk_start: float,
        chunk_end: float,
        missing_values: int | Rate,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(missing_values, int) and missing_values < 0:
            raise ValueError(
                f"missing_values should be a non-negative integer or a Rate,"
                f" got '{missing_values}'"
            )
        self.chunk_start = chunk_start
        self.chunk_end = chunk_end
        self.missing_values = missing_values
        self.random_state = random_state

    def create_missing_data(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
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
        rng = np.random.default_rng(self.random_state)
        missing_indexes = rng.choice(
            chunk_indexes.to_numpy(),
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
        return isinstance(value, Mar) and (
            self.chunk_start,
            self.chunk_end,
            self.missing_values,
            self.random_state,
        ) == (
            value.chunk_start,
            value.chunk_end,
            value.missing_values,
            value.random_state,
        )

    def __hash__(self) -> int:
        return hash(Mar)


class DataPreparator[MLT, DT: DatasetDescriptor](ABC):
    """Base class for data preparators.

    Parameters
    ----------
    dataset_descriptor : DT
        Descriptor of the dataset to prepare.
    missing_generator : MissingDataGenerator
        Generator injecting missing values in the prepared data.
    sample_size : int | None
        Number of rows to use; None uses all available rows.
    random_state : int | None
        Seed for the stochastic steps of data preparation (row sampling,
        synthetic-data generation); None yields nondeterministic draws.
        Missing-data generators carry their own random_state. Orchestration
        must derive separate seeds with split_random_state when seeding both.
    """

    def __init__(
        self,
        dataset_descriptor: DT,
        missing_generator: MissingDataGenerator,
        sample_size: int | None,
        random_state: int | None = None,
    ):
        self.dataset_descriptor = dataset_descriptor
        self.missing_generator = missing_generator
        self.sample_size = sample_size
        self.random_state = random_state

        self.missing_indexes: np.ndarray
        self.actual_values: pd.Series

    @property
    def missing_type(self) -> MissingDataGenerator:
        """Compatibility alias for missing_generator."""
        return self.missing_generator

    @missing_type.setter
    def missing_type(self, value: MissingDataGenerator) -> None:
        self.missing_generator = value

    @abstractmethod
    def prepare_data(self) -> MLT:
        raise NotImplementedError

    def __str__(self) -> str:
        return (
            f"{self.dataset_descriptor} \n"
            f"\tmissing generator: {self.missing_generator} \n"
            f"\tsample size: {self.sample_size}"
        )


class ExcelDataPreparator(DataPreparator[DataFrameMLData, ExcelDatasetDescriptor]):
    def __init__(
        self,
        dataset_descriptor: ExcelDatasetDescriptor,
        missing_generator: MissingDataGenerator,
        sample_size: int | None = None,
        columns_to_keep: list[str] | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            dataset_descriptor=dataset_descriptor,
            missing_generator=missing_generator,
            sample_size=sample_size,
            random_state=random_state,
        )
        self.columns_to_keep = columns_to_keep if columns_to_keep is not None else []

    def prepare_data(self) -> DataFrameMLData:
        if self.sample_size is None:
            df = pd.read_excel(
                io=self.dataset_descriptor.path,
                sheet_name=self.dataset_descriptor.sheet_name,
                usecols=[
                    self.dataset_descriptor.input_column,
                    self.dataset_descriptor.target_column,
                    *self.columns_to_keep,
                ],
            )
        else:
            df = pd.read_excel(
                io=self.dataset_descriptor.path,
                sheet_name=self.dataset_descriptor.sheet_name,
                usecols=[
                    self.dataset_descriptor.input_column,
                    self.dataset_descriptor.target_column,
                    *self.columns_to_keep,
                ],
            ).sample(self.sample_size, random_state=self.random_state)

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
            self.missing_generator.create_missing_data(df)
        )

        return DataFrameMLData(
            df, DatasetDescriptor("X", "Y", other_columns=self.columns_to_keep)
        )


class DataGenerator(DataPreparator[DataFrameMLData, DatasetDescriptor]):
    """Synthetic data generator.

    Parameters
    ----------
    linear_interpolation_ratio : float
        Slope applied to X to build the linear geometry target.
    sample_size : int
        Number of generated rows.
    missing_generator : MissingDataGenerator
        Generator injecting missing values in the generated data.
    geometry_type : {'linear', 'ring'}
        Shape of the generated distribution.
    dataset_descriptor : DatasetDescriptor | None
        Column names for the generated data; defaults to X and Y.
    excel_path : str | None
        Stored for compatibility; does not write a workbook.
    random_state : int | None
        Seed for reproducible synthetic-data generation; each call to
        prepare_data replays the stream seeded with it. None
        yields nondeterministic draws.
    """

    def __init__(
        self,
        linear_interpolation_ratio: float,
        sample_size: int,
        missing_generator: MissingDataGenerator,
        geometry_type: GeometryType = "linear",
        dataset_descriptor: DatasetDescriptor | None = None,
        excel_path: str | None = None,
        *,
        random_state: int | None = None,
    ):
        if geometry_type not in GEOMETRY_TYPES:
            raise ValueError(
                f"geometry_type must be one of: {', '.join(repr(g) for g in GEOMETRY_TYPES)},"
                f" got '{geometry_type}'"
            )
        super().__init__(
            dataset_descriptor=(
                DatasetDescriptor("X", "Y")
                if dataset_descriptor is None
                else dataset_descriptor
            ),
            missing_generator=missing_generator,
            sample_size=sample_size,
            random_state=random_state,
        )
        self.linear_interpolation_ratio = linear_interpolation_ratio
        self.geometry_type: GeometryType = geometry_type
        self.excel_path = excel_path
        # Narrow the sample_size attribute to int: DataGenerator requires an
        # explicit sample size, unlike ExcelDataPreparator which accepts None.
        self.sample_size: int

    def build_synthetic_data(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        match self.geometry_type:
            case "linear":
                x = np.linspace(-2, 2, self.sample_size)
                epsilon = rng.chisquare(2, self.sample_size)
                y = self.linear_interpolation_ratio * x + epsilon

            case "ring":
                theta = np.linspace(0, 2 * np.pi, self.sample_size)
                epsilon = rng.normal(0, 0.1, self.sample_size)
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
        descriptor = self.dataset_descriptor
        create_missing_data = self.missing_generator.create_missing_data
        # Built-ins require X/Y; custom implementations keep descriptor names.
        normalize_columns = (descriptor.input_column, descriptor.target_column) != (
            "X",
            "Y",
        ) and getattr(create_missing_data, "__func__", None) in (
            Mcar.create_missing_data,
            Mar.create_missing_data,
        )
        if normalize_columns:
            df = df.rename(
                columns={descriptor.input_column: "X", descriptor.target_column: "Y"}
            )
        self.missing_indexes, self.actual_values = create_missing_data(df)
        if normalize_columns:
            df = df.rename(
                columns={"X": descriptor.input_column, "Y": descriptor.target_column}
            )
            if self.actual_values.name == "Y":
                self.actual_values = self.actual_values.rename(descriptor.target_column)
        return DataFrameMLData(df, descriptor)
