from typing import Literal, Optional, Union, overload

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data_preparation.data_description import DataFrameMLData, MLSets
from src.imputation import UncertaintyImputer
from src.imputation.imputer import BoundsPerPercentile

ImputationStrategy = Literal["sample", "mean", "median"]
KNNAlgorithm = Literal["auto", "ball_tree", "kd_tree", "brute"]
ScalingStrategy = Literal["none", "standardization", "normalization"]
KOptimalMethod = Literal["heuristic", "cv"]
KNNWeights = Literal["uniform", "distance"]


class KnnSampler(UncertaintyImputer):
    """k-Nearest Neighbors based sampler for imputing missing target values.

    This imputer samples or aggregates target values from the k nearest neighbors
    based on a single-feature distance. Optional empirical bounds are
    produced by computing percentile intervals over each neighborhood.

    Distinct scaling strategies can be specified for:
      - the internal optimal k selection (scaling_optimal_k)
      - model fitting & imputation (scaling_fit)

    Parameters
    ----------
    ml_data : DataFrameMLData
        Wrapped dataframe + descriptor.
    compute_bounds : bool
        Whether to compute percentile-based empirical bounds for each imputed row.
    lower_percentiles / upper_percentiles : list[float] | None
        Percentile pairs defining empirical bounds; defaults to [5],[95].
    n_neighbors : int | None
        If provided, use this k; otherwise an internal heuristic selects it.
    random_state : int | None
        RNG seed for reproducible sampling (only for strategy='sample').
    strategy : {'sample','mean','median'}
        How to produce the final imputed value from neighbors.
    algorithm : {'auto','ball_tree','kd_tree','brute'}
        KNN search algorithm.
    scaling_optimal_k : {'none','standardization','normalization'}
        Scaling applied only for the internal heuristic choosing k (default 'standardization').
    scaling_fit : {'none','standardization','normalization'}
        Scaling applied to data used to fit the final KNN and during imputation (default 'none').
    optimal_k_method : {'heuristic','cv'}
        Method for selecting optimal k: 'heuristic' uses adjusted MSE, 'cv' uses cross-validation (default 'heuristic').
    weights : {'uniform','distance'}
        Weighting strategy passed to scikit-learn KNeighborsRegressor (default 'uniform').
    cv_folds : int
        Number of folds for cross-validation when optimal_k_method='cv' (default 3).

    Notes
    -----
    Dual Scaling Strategy:
        The separation of scaling_optimal_k and scaling_fit allows for optimal k selection
        in a normalized space (better distance comparisons) while preserving the original
        scale for imputed values (better interpretability). Note that the optimal k found
        in the scaled space may not be optimal in the unscaled space - this is an accepted
        trade-off for practical flexibility.

    Empirical Bounds:
        The computed bounds are local neighborhood quantiles, not statistical confidence
        or prediction intervals. They provide empirical uncertainty estimates based solely
        on the distribution of neighbor target values.
    """

    @overload
    def __init__(
        self,
        ml_data: DataFrameMLData,
        compute_bounds: Literal[False],
        lower_percentiles: None = None,
        upper_percentiles: None = None,
        n_neighbors: Optional[int] = None,
        random_state: Optional[int] = None,
        strategy: ImputationStrategy = "sample",
        algorithm: KNNAlgorithm = "kd_tree",
        scaling_optimal_k: ScalingStrategy = "standardization",
        scaling_fit: ScalingStrategy = "none",
        optimal_k_method: KOptimalMethod = "heuristic",
        weights: KNNWeights = "uniform",
        cv_folds: int = 3,
    ): ...

    @overload
    def __init__(
        self,
        ml_data: DataFrameMLData,
        compute_bounds: Literal[True],
        lower_percentiles: list[float],
        upper_percentiles: list[float],
        n_neighbors: Optional[int] = None,
        random_state: Optional[int] = None,
        strategy: ImputationStrategy = "sample",
        algorithm: KNNAlgorithm = "kd_tree",
        scaling_optimal_k: ScalingStrategy = "standardization",
        scaling_fit: ScalingStrategy = "none",
        optimal_k_method: KOptimalMethod = "heuristic",
        weights: KNNWeights = "uniform",
        cv_folds: int = 3,
    ): ...

    def __init__(
        self,
        ml_data: DataFrameMLData,
        compute_bounds: bool,
        lower_percentiles: list[float] | None = None,
        upper_percentiles: list[float] | None = None,
        n_neighbors: Optional[int] = None,
        random_state: Optional[int] = None,
        strategy: ImputationStrategy = "sample",
        algorithm: KNNAlgorithm = "kd_tree",
        scaling_optimal_k: ScalingStrategy = "standardization",
        scaling_fit: ScalingStrategy = "none",
        optimal_k_method: KOptimalMethod = "heuristic",
        weights: KNNWeights = "uniform",
        cv_folds: int = 3,
    ):
        # Defaults percentiles
        lower_percentiles = lower_percentiles or [5]
        upper_percentiles = upper_percentiles or [95]

        if n_neighbors is not None and n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if strategy not in ("sample", "mean", "median"):
            raise ValueError("strategy must be one of: 'sample', 'mean', 'median'")
        if algorithm not in ("auto", "ball_tree", "kd_tree", "brute"):
            raise ValueError(
                "algorithm must be one of: 'auto','ball_tree','kd_tree','brute'"
            )
        for label, value in (
            ("scaling_optimal_k", scaling_optimal_k),
            ("scaling_fit", scaling_fit),
        ):
            if value not in ("none", "standardization", "normalization"):
                raise ValueError(
                    f"{label} must be one of: 'none','standardization','normalization'"
                )
        if optimal_k_method not in ("heuristic", "cv"):
            raise ValueError("optimal_k_method must be one of: 'heuristic','cv'")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be one of: 'uniform','distance'")
        if cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")

        super().__init__(
            ml_data=ml_data,
            lower_percentiles=lower_percentiles,
            upper_percentiles=upper_percentiles,
            compute_bounds=compute_bounds,
        )
        self.optimal_k: int | None = n_neighbors
        self.knn: KNeighborsRegressor | None = None
        self.scaler: Union[StandardScaler, MinMaxScaler, None] = None
        self.random_state = random_state
        self.strategy: ImputationStrategy = strategy
        self.algorithm: KNNAlgorithm = algorithm
        self.scaling_optimal_k: ScalingStrategy = scaling_optimal_k
        self.scaling_fit: ScalingStrategy = scaling_fit
        self.optimal_k_method: KOptimalMethod = optimal_k_method
        self.weights: KNNWeights = weights
        self.cv_folds: int = cv_folds
        self._rng: np.random.Generator | None = None

    def _init_rng(self) -> None:
        if self.strategy == "sample":
            self._rng = np.random.default_rng(self.random_state)
        else:
            self._rng = None

    def _optimal_k_scaling(self, X: pd.DataFrame) -> np.ndarray:
        """Return transformed X for k selection."""
        if self.scaling_optimal_k == "standardization":
            return StandardScaler().fit_transform(X)
        if self.scaling_optimal_k == "normalization":
            return MinMaxScaler().fit_transform(X)
        return X.to_numpy()

    def _fit_scaler(self, X: pd.DataFrame):
        if self.scaling_fit == "standardization":
            self.scaler = StandardScaler().fit(X)
        elif self.scaling_fit == "normalization":
            self.scaler = MinMaxScaler().fit(X)
        else:
            self.scaler = None

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            return X.to_numpy()
        return self.scaler.transform(X)

    def find_optimal_k(self, train_sets: MLSets) -> int:
        x_train, y_train = train_sets.x, train_sets.y
        x_scaled = self._optimal_k_scaling(x_train)
        n = len(x_train)
        max_k = min(int(np.sqrt(n)), n - 1)
        if max_k <= 1:
            return 1
        k_values: list[int] = list(range(1, max_k + 1))
        scores: list[float] = []
        for k in k_values:
            regressor = KNeighborsRegressor(
                n_neighbors=k, algorithm=self.algorithm, weights=self.weights
            )
            regressor.fit(x_scaled, y_train)
            y_pred = regressor.predict(x_scaled)
            mse = np.mean((y_train - y_pred) ** 2)
            adjusted_score = mse * ((k + 1) ** 2 / (k**2))
            scores.append(adjusted_score)
        return k_values[np.argmin(scores)]

    def find_optimal_k_kfold(
        self, train_sets: MLSets, cv_folds: int | None = None
    ) -> int:
        """K-fold cross-validation for optimal k selection."""
        x_train, y_train = train_sets.x, train_sets.y
        x_scaled = self._optimal_k_scaling(x_train)
        n = len(x_train)
        max_k = min(int(np.sqrt(n)), n - 1)

        if max_k <= 1:
            return 1

        k_values = list(range(1, max_k + 1))
        cv_scores = []
        folds = cv_folds or self.cv_folds
        random_state = self.random_state if self.random_state is not None else 42

        for k in k_values:
            fold_scores = []
            kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)

            for train_idx, val_idx in kfold.split(x_scaled):
                knn = KNeighborsRegressor(
                    n_neighbors=k, algorithm=self.algorithm, weights=self.weights
                )
                knn.fit(x_scaled[train_idx], y_train.iloc[train_idx])
                y_pred = knn.predict(x_scaled[val_idx])
                mse = np.mean((y_train.iloc[val_idx] - y_pred) ** 2)
                fold_scores.append(mse)

            cv_scores.append(np.mean(fold_scores))

        return k_values[np.argmin(cv_scores)]

    def fit(self):
        nona_sets = self.ml_data.nona_sets()
        if self.optimal_k is None:
            if self.optimal_k_method == "cv":
                self.optimal_k = self.find_optimal_k_kfold(nona_sets)
            else:
                self.optimal_k = self.find_optimal_k(nona_sets)
        if self.optimal_k is None or self.optimal_k < 1:
            self.optimal_k = 1
        if self.optimal_k > len(nona_sets.x):
            self.optimal_k = len(nona_sets.x)
        self._fit_scaler(nona_sets.x)
        X_train_data = self._transform(nona_sets.x)
        self.knn = KNeighborsRegressor(
            n_neighbors=self.optimal_k, algorithm=self.algorithm, weights=self.weights
        )
        self.knn.fit(X_train_data, nona_sets.y)
        self._init_rng()

    def _execute(self) -> pd.DataFrame:
        imputed_df = self.ml_data.df
        imputed_target_values, self.bounds = self.impute_for_dataset(
            imputed_df, self.ml_data.nona_sets().y
        )
        imputed_df.loc[
            imputed_target_values.index, self.ml_data.dataset_descriptor.target_column
        ] = imputed_target_values
        return imputed_df

    def impute_for_dataset(
        self, df: pd.DataFrame, y_train: pd.Series
    ) -> tuple[pd.Series, BoundsPerPercentile]:
        """Impute missing target values and optionally compute empirical bounds.

        The bounds are empirical quantiles of the k nearest neighbors' target values.

        Returns
        -------
        imputed_target_values : pd.Series
            Imputed target values indexed by original dataframe indices (only missing rows).
        bounds : dict
            Mapping (lower_percentile, upper_percentile) -> (list[lower_values], list[upper_values]).
            Lists contain empirical quantiles from neighbor target values, aligned with imputation order.
        """
        if self.knn is None:
            raise ValueError("Model not fitted: call fit() before imputation")
        imputed_series = pd.Series(dtype="float64")
        bounds: BoundsPerPercentile = {
            (lp, up): ([], [])
            for lp, up in zip(
                self.lower_percentiles, self.upper_percentiles, strict=True
            )
        }
        missing_mask = df[self.ml_data.dataset_descriptor.target_column].isna()
        missing_rows = df.loc[missing_mask]
        if missing_rows.empty:
            return imputed_series, bounds
        X_missing = missing_rows[[self.ml_data.dataset_descriptor.input_column]]
        X_missing_array = self._transform(X_missing)
        _, ind = self.knn.kneighbors(X_missing_array)
        y_array = y_train.to_numpy()
        y_neighbors = y_array[ind]
        M, k = y_neighbors.shape
        if self.compute_bounds:
            for lp, up in zip(
                self.lower_percentiles, self.upper_percentiles, strict=True
            ):
                lower_percentiles_array = np.percentile(y_neighbors, lp, axis=1)
                upper_percentiles_array = np.percentile(y_neighbors, up, axis=1)
                bounds[(lp, up)] = (
                    lower_percentiles_array.astype(float).tolist(),
                    upper_percentiles_array.astype(float).tolist(),
                )
        if self.strategy == "sample":
            rng = self._rng or np.random.default_rng()
            random_columns = rng.integers(0, k, size=M)
            imputed_values = y_neighbors[np.arange(M), random_columns]
        elif self.strategy == "mean":
            imputed_values = np.mean(y_neighbors, axis=1)
        else:
            imputed_values = np.median(y_neighbors, axis=1)
        imputed_series = pd.Series(
            imputed_values, index=missing_rows.index, dtype="float64"
        )
        return imputed_series, bounds
