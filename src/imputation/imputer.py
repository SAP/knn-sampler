from abc import ABC, abstractmethod
from typing import overload

import pandas as pd

from src.data_preparation.data_description import DataFrameMLData


class Imputer(ABC):
    """Abstract base class for single-feature target imputers.

    Responsibilities
    ----------------
    - Hold a reference to prepared ML data (features + target descriptor).
    - Provide a uniform interface: fit() then execute()/ _execute().
    - Enforce a post-condition: returned DataFrame must not contain NaN in the target.

    Execution Randomness
    --------------------
    Implementations may optionally use the ``random_state`` argument of ``execute`` /
    ``_execute`` to control stochastic sampling (if applicable). Deterministic
    imputers ignore it.

    Methods to implement
    --------------------
    fit(): train / prepare any internal model artifacts.
    _execute(random_state): perform the actual imputation and return a DataFrame.
    """

    def __init__(
        self,
        ml_data: DataFrameMLData,
    ):
        self.ml_data = ml_data

    def get_name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def fit(self) -> None:
        """Fit / prepare internal state."""
        raise NotImplementedError

    @overload
    def execute(self) -> pd.DataFrame: ...

    @overload
    def execute(self, random_state: int | None) -> pd.DataFrame: ...

    def execute(self, random_state: int | None = None) -> pd.DataFrame:
        """Run imputation and validate result.

        Parameters
        ----------
        random_state : int | None
            Optional seed forwarded to the implementation for stochastic imputers.

        Returns
        -------
        pd.DataFrame
            Copy / view of the underlying data with target values imputed.

        Raises
        ------
        ValueError
            If the resulting DataFrame still contains NaN in the target column.
        """
        result = self._execute(random_state)
        target_col = self.ml_data.dataset_descriptor.target_column
        if result[target_col].isna().sum() > 0:
            raise ValueError(
                "Imputation error: DataFrame contains NaN values in target column"
            )
        return result

    @overload
    def _execute(self) -> pd.DataFrame: ...

    @overload
    def _execute(self, random_state: int | None) -> pd.DataFrame: ...

    @abstractmethod
    def _execute(self, random_state: int | None = None) -> pd.DataFrame:
        """Concrete imputation logic."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.get_name()


# Couple of percentile to couple of bounds (lower then upper)
type BoundsPerPercentile = dict[tuple[float, float], tuple[list[float], list[float]]]


class UncertaintyImputer(Imputer):
    """Base imputer supporting empirical percentile bounds for imputed targets.

    Bounds are computed per-row from neighborhood target distributions and
    stored in ``self.bounds`` as:
        {(lower_pct, upper_pct): (list_of_lower_values, list_of_upper_values)}

    Attributes
    ----------
    bounds : dict
        Mapping percentile pair -> (lower_values, upper_values) lists.
    lower_percentiles / upper_percentiles : list[float]
        Matched percentile definitions (same length, lower < upper per pair).
    compute_bounds : bool
        Flag controlling whether bounds are calculated during imputation.
    """

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
        """Validate percentile pair lists.

        Raises
        ------
        ValueError
            If length mismatch or invalid (lower >= upper) or out of [0,100].
        """
        if len(lower_percentiles) != len(upper_percentiles):
            raise ValueError(
                "lower_percentiles and upper_percentiles must have equal length"
            )
        for lp, up in zip(lower_percentiles, upper_percentiles, strict=True):
            if not (0.0 <= lp < up <= 100.0):
                raise ValueError(
                    f"Invalid percentile pair ({lp}, {up}). Require 0 <= lower < upper <= 100."
                )
