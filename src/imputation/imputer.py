from abc import ABC, abstractmethod

import pandas as pd

from src.data_preparation.data_description import DataFrameMLData


class Imputer(ABC):
    def __init__(
        self,
        ml_data: DataFrameMLData,
    ):
        self.ml_data = ml_data

    def get_name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError

    def execute(self) -> pd.DataFrame:
        result = self._execute()
        if result.Y.isna().sum() > 0:
            raise ValueError("Imputation error: DataFrame contains NaN values")
        return result

    @abstractmethod
    def _execute(self) -> pd.DataFrame:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.get_name()


# Couple of percentile to couple of bounds (lower then upper)
type BoundsPerPercentile = dict[tuple[float, float], tuple[list[float], list[float]]]


class UncertaintyImputer(Imputer):
    bounds: BoundsPerPercentile
    lower_percentiles: list[float]
    upper_percentiles: list[float]

    def __init__(
        self,
        ml_data: DataFrameMLData,
        lower_percentile: list[float],
        upper_percentile: list[float],
        compute_bounds: bool = True,
    ):
        super().__init__(ml_data=ml_data)
        if (ll := len(lower_percentile)) != (ul := len(upper_percentile)):
            raise ValueError(
                f"lower and upper bounds length are expected to be equal, got {ll} and {ul}"
            )
        self.bounds = {}
        self.lower_percentiles = lower_percentile
        self.upper_percentiles = upper_percentile
        self.compute_bounds: bool = compute_bounds
