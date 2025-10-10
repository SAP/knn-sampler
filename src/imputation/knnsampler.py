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
    """k-NN sampler for missing target value imputation.

    Imputes missing target values by sampling from k nearest neighbors based on
    single input feature distance. Supports mean/median aggregation and optional
    empirical uncertainty bounds computed as percentile intervals over neighborhoods.

    Parameters
    ----------
    ml_data : DataFrameMLData
        Prepared ML data wrapper.
    compute_bounds : bool
        Whether to compute empirical percentile bounds.
    lower_percentiles : list[float] | None
        Lower percentiles for bounds, defaults to [5].
    upper_percentiles : list[float] | None
        Upper percentiles for bounds, defaults to [95].
    n_neighbors : int | None
        Fixed k value. If None, selected via optimal_k_method.
    strategy : {'sample', 'mean', 'median'}
        Aggregation strategy for neighbor targets.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
        KNN search algorithm.
    weights : {'uniform', 'distance'}
        Neighbor weighting strategy.
    optimal_k_method : {'heuristic', 'cv'}
        Method for k selection when n_neighbors is None.
    optimal_k_random_state : int | None
        Seed for K-Fold cross-validation during k selection.
    optimal_k_cv_folds : int
        Number of CV folds, must be >= 2.
    scaling_optimal_k : {'none', 'standardization', 'normalization'}
        Scaling applied during k selection phase.
    scaling_fit : {'none', 'standardization', 'normalization'}
        Scaling applied for final fitting and imputation.

    Notes
    -----
    Execution randomness is controlled via execute(random_state=...).
    Cross-validation randomness is controlled via optimal_k_random_state.
    For strategy='sample', k is automatically increased to 2 when possible.
    """

    @overload
    def __init__(
        self,
        ml_data: DataFrameMLData,
        compute_bounds: Literal[False],
        lower_percentiles: None = None,
        upper_percentiles: None = None,
        n_neighbors: Optional[int] = None,
        strategy: ImputationStrategy = "sample",
        algorithm: KNNAlgorithm = "kd_tree",
        weights: KNNWeights = "uniform",
        optimal_k_method: KOptimalMethod = "heuristic",
        optimal_k_random_state: Optional[int] = None,
        optimal_k_cv_folds: int = 3,
        scaling_optimal_k: ScalingStrategy = "standardization",
        scaling_fit: ScalingStrategy = "none",
    ): ...

    @overload
    def __init__(
        self,
        ml_data: DataFrameMLData,
        compute_bounds: Literal[True],
        lower_percentiles: list[float],
        upper_percentiles: list[float],
        n_neighbors: Optional[int] = None,
        strategy: ImputationStrategy = "sample",
        algorithm: KNNAlgorithm = "kd_tree",
        weights: KNNWeights = "uniform",
        optimal_k_method: KOptimalMethod = "heuristic",
        optimal_k_random_state: Optional[int] = None,
        optimal_k_cv_folds: int = 3,
        scaling_optimal_k: ScalingStrategy = "standardization",
        scaling_fit: ScalingStrategy = "none",
    ): ...

    def __init__(
        self,
        ml_data: DataFrameMLData,
        compute_bounds: bool,
        lower_percentiles: list[float] | None = None,
        upper_percentiles: list[float] | None = None,
        n_neighbors: Optional[int] = None,
        strategy: ImputationStrategy = "sample",
        algorithm: KNNAlgorithm = "kd_tree",
        weights: KNNWeights = "uniform",
        optimal_k_method: KOptimalMethod = "heuristic",
        optimal_k_random_state: Optional[int] = None,
        optimal_k_cv_folds: int = 3,
        scaling_optimal_k: ScalingStrategy = "standardization",
        scaling_fit: ScalingStrategy = "none",
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
        if optimal_k_cv_folds < 2:
            raise ValueError("optimal_k_cv_folds must be >= 2")

        super().__init__(
            ml_data=ml_data,
            lower_percentiles=lower_percentiles,
            upper_percentiles=upper_percentiles,
            compute_bounds=compute_bounds,
        )
        self.optimal_k: int | None = n_neighbors
        self.knn: KNeighborsRegressor | None = None
        self.scaler: Union[StandardScaler, MinMaxScaler, None] = None
        self.strategy: ImputationStrategy = strategy
        self.algorithm: KNNAlgorithm = algorithm
        self.weights: KNNWeights = weights
        self.optimal_k_method: KOptimalMethod = optimal_k_method
        self.optimal_k_random_state: Optional[int] = optimal_k_random_state
        self.optimal_k_cv_folds: int = optimal_k_cv_folds
        self.scaling_optimal_k: ScalingStrategy = scaling_optimal_k
        self.scaling_fit: ScalingStrategy = scaling_fit
        self._rng: np.random.Generator | None = None

    def _init_rng(self, random_state: Optional[int] = None) -> None:
        if self.strategy == "sample":
            seed = random_state if isinstance(random_state, int) else None
            self._rng = np.random.default_rng(seed)
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

    def _calculate_mse(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate mean squared error between true and predicted values."""
        return np.mean((y_true - y_pred) ** 2)

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
            mse = self._calculate_mse(y_train, y_pred)
            adjusted_score = mse * ((k + 1) ** 2 / (k**2))
            scores.append(adjusted_score)
        return k_values[np.argmin(scores)]

    def find_optimal_k_kfold(self, train_sets: MLSets) -> int:
        """K-fold cross-validation for optimal k selection."""
        x_train, y_train = train_sets.x, train_sets.y
        x_scaled = self._optimal_k_scaling(x_train)
        n = len(x_train)
        max_k = min(int(np.sqrt(n)), n - 1)

        if max_k <= 1:
            return 1

        k_values = list(range(1, max_k + 1))
        cv_scores = []

        for k in k_values:
            fold_scores = []
            kfold = KFold(
                n_splits=self.optimal_k_cv_folds,
                shuffle=True,
                random_state=self.optimal_k_random_state,
            )

            for train_idx, test_idx in kfold.split(x_scaled):
                knn = KNeighborsRegressor(
                    n_neighbors=k, algorithm=self.algorithm, weights=self.weights
                )
                knn.fit(x_scaled[train_idx], y_train.iloc[train_idx])
                y_pred = knn.predict(x_scaled[test_idx])
                mse = self._calculate_mse(y_train.iloc[test_idx], y_pred)
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
        if self.strategy == "sample" and self.optimal_k == 1 and len(nona_sets.x) >= 2:
            self.optimal_k = 2
        self._fit_scaler(nona_sets.x)
        X_train_data = self._transform(nona_sets.x)
        self.knn = KNeighborsRegressor(
            n_neighbors=self.optimal_k, algorithm=self.algorithm, weights=self.weights
        )
        self.knn.fit(X_train_data, nona_sets.y)

    def _execute(self, random_state: int | None = None) -> pd.DataFrame:
        imputed_df = self.ml_data.df
        imputed_target_values, self.bounds = self.impute_for_dataset(
            imputed_df, self.ml_data.nona_sets().y, random_state
        )
        imputed_df.loc[
            imputed_target_values.index, self.ml_data.dataset_descriptor.target_column
        ] = imputed_target_values
        return imputed_df

    def impute_for_dataset(
        self, df: pd.DataFrame, y_train: pd.Series, random_state: Optional[int] = None
    ) -> tuple[pd.Series, BoundsPerPercentile]:
        """Impute missing target values and compute empirical bounds.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing missing target values to impute.
        y_train : pd.Series
            Training target values for neighborhood lookup.
        random_state : int | None
            Seed for reproducible sampling when strategy='sample'.

        Returns
        -------
        imputed_target_values : pd.Series
            Imputed values indexed by original dataframe row indices.
        bounds : BoundsPerPercentile
            Empirical percentile bounds from neighbor target distributions.
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
        _, indexes = self.knn.kneighbors(X_missing_array)
        y_array = y_train.to_numpy()
        y_neighbors = y_array[indexes]
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
            self._init_rng(random_state)
            rng = self._rng or np.random.default_rng(random_state)
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
