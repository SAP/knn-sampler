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
        lower_percentiles: list[float],
        upper_percentiles: list[float],
        compute_bounds: bool = True,
    ):
        super().__init__(ml_data=ml_data)
        UncertaintyImputer._validate_percentiles(lower_percentiles, upper_percentiles)
        self.bounds = {}
        self.lower_percentiles = lower_percentiles
        self.upper_percentiles = upper_percentiles
        self.compute_bounds: bool = compute_bounds

    @staticmethod
    def _validate_percentiles(
        lower_percentiles: list[float], upper_percentiles: list[float]
    ) -> None:
        if len(lower_percentiles) != len(upper_percentiles):
            raise ValueError(
                "lower_percentiles and upper_percentiles must have equal length"
            )
        for lp, up in zip(lower_percentiles, upper_percentiles, strict=False):
            if not (0.0 <= lp < up <= 100.0):
                raise ValueError(
                    f"Invalid percentile pair ({lp}, {up}). Require 0 <= lower < upper <= 100."
                )
