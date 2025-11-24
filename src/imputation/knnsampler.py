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
KOptimalMethod = Literal["loo_penalized", "kfold"]
KNNWeights = Literal["uniform", "distance"]


def _validate_option(name: str, value: str, allowed: tuple[str, ...]) -> None:
    """Validate that `value` is one of `allowed` and raise ValueError with
    a consistent message if not.

    This centralizes Literal-like runtime checks and keeps error messages
    homogeneous across the module.

    Parameters
    ----------
    name : str
        Name of the parameter being validated, used in error messages.
    value : str
        The value to validate against allowed options.
    allowed : tuple[str, ...]
        Tuple of allowed string values.

    Raises
    ------
    ValueError
        If `value` is not in `allowed` tuple.
    """
    if value not in allowed:
        allowed_str = ", ".join(f"'{a}'" for a in allowed)
        raise ValueError(f"{name} must be one of: {allowed_str}")


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
    optimal_k_method : {'loo_penalized', 'kfold'}
         Method used when n_neighbors is None:
         - 'loo_penalized': leave-one-out penalized MSE criterion.
         - 'kfold': K-fold cross-validation MSE.
    optimal_k_random_state : int | None
        Seed for K-Fold splitting when optimal_k_method='kfold'.
    optimal_k_cv_folds : int
        Number of CV folds (>=2) for optimal_k_method='kfold'.
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
        optimal_k_method: KOptimalMethod = "loo_penalized",
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
        optimal_k_method: KOptimalMethod = "loo_penalized",
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
        optimal_k_method: KOptimalMethod = "loo_penalized",
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

        _validate_option("strategy", strategy, ("sample", "mean", "median"))
        _validate_option(
            "algorithm", algorithm, ("auto", "ball_tree", "kd_tree", "brute")
        )
        for label, value in (
            ("scaling_optimal_k", scaling_optimal_k),
            ("scaling_fit", scaling_fit),
        ):
            _validate_option(label, value, ("none", "standardization", "normalization"))
        _validate_option(
            "optimal_k_method", optimal_k_method, ("loo_penalized", "kfold")
        )
        _validate_option("weights", weights, ("uniform", "distance"))
        if optimal_k_cv_folds < 2:
            raise ValueError(
                f"optimal_k_cv_folds must be >= 2, got {optimal_k_cv_folds}"
            )

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
        """Initialize random number generator for sampling strategy.

        Parameters
        ----------
        random_state : int, optional
            Seed for reproducible random number generation. Only used when
            strategy is 'sample'.
        """
        if self.strategy == "sample":
            seed = random_state if isinstance(random_state, int) else None
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = None

    def _optimal_k_scaling(self, X: pd.DataFrame) -> np.ndarray:
        """Apply scaling transformation for k selection phase.

        Parameters
        ----------
        X : pd.DataFrame
            Input features to transform.

        Returns
        -------
        np.ndarray
            Scaled feature matrix according to scaling_optimal_k strategy.
        """
        if self.scaling_optimal_k == "standardization":
            return StandardScaler().fit_transform(X)
        if self.scaling_optimal_k == "normalization":
            return MinMaxScaler().fit_transform(X)
        return X.to_numpy()

    def _fit_scaler(self, X: pd.DataFrame) -> None:
        """Fit scaling transformation for final imputation phase.

        Parameters
        ----------
        X : pd.DataFrame
            Training features to fit the scaler on.

        Raises
        ------
        ValueError
            If X is an empty DataFrame.
        """
        if X.empty:
            raise ValueError("Cannot fit scaler on empty DataFrame")

        if self.scaling_fit == "standardization":
            self.scaler = StandardScaler().fit(X)
        elif self.scaling_fit == "normalization":
            self.scaler = MinMaxScaler().fit(X)
        else:
            self.scaler = None

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.

        Returns
        -------
        np.ndarray
            Transformed feature matrix, or original array if no scaling.
        """
        if self.scaler is None:
            return X.to_numpy()
        return self.scaler.transform(X)

    def _calculate_mse(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate mean squared error between true and predicted values.

        Parameters
        ----------
        y_true : pd.Series
            Ground truth target values.
        y_pred : np.ndarray
            Predicted target values.

        Returns
        -------
        float
            Mean squared error.
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def _get_k_bounds(self, n_samples: int) -> tuple[int, int]:
        """Compute (min_k, max_k) bounds for candidates k selection.

        Rules:
        - If n_samples <= 2: return (1, 1).
        - min_k is always 1 otherwise.
        - Heuristic upper candidate: floor(sqrt(n_samples)).
        - Structural cap:
          * 'loo_penalized': n_samples - 1 (not including self).
          * 'kfold': n_samples - ceil(n_samples / n_folds) (smallest training fold size; ceil(n_samples / n_folds) is the largest test fold size).
        - max_k = min(heuristic_upper, structural_cap); enforce max_k >= min_k.

        Parameters
        ----------
        n_samples : int
            Total number of rows in the dataset used for k search.

        Returns
        -------
        tuple[int, int]
            (min_k, max_k) with both >= 1.
        """
        if n_samples <= 2:
            return 1, 1
        min_k = 1
        heuristic_max_k = int(np.sqrt(n_samples))
        structural_max_k = np.inf
        if self.optimal_k_method == "loo_penalized":
            structural_max_k = n_samples - 1
        elif self.optimal_k_method == "kfold":
            folds = self.optimal_k_cv_folds
            structural_max_k = n_samples - int(np.ceil(n_samples / folds))
        max_k = min(heuristic_max_k, structural_max_k)
        if max_k < min_k:
            max_k = min_k
        return min_k, max_k

    def find_optimal_k(self, train_sets: MLSets) -> int:
        """Select optimal k via penalized leave-one-out MSE.

        For each candidate k, forms leave-one-out predictions using neighborhoods
        excluding the sample itself. Applies multiplicative penalty ((k+1)^2 / k^2)
        favoring smaller k when MSE differences are marginal. Returns k with
        minimal penalized MSE.

        Parameters
        ----------
        train_sets : MLSets
            Training data (features and target) without missing target values.

        Returns
        -------
        int
            Optimal number of neighbors (>=1).
        """
        n_samples = len(self.ml_data.df)
        x_train, y_train = train_sets.x, train_sets.y
        x_scaled = self._optimal_k_scaling(x_train)
        min_k, max_k = self._get_k_bounds(n_samples)
        if max_k <= min_k:
            return min_k

        # Fit one KNN model including self with max_k + 1 to extract neighbor structure
        knn = KNeighborsRegressor(
            n_neighbors=max_k + 1, algorithm=self.algorithm, weights=self.weights
        )
        knn.fit(x_scaled, y_train)
        distances, indices = knn.kneighbors(x_scaled, return_distance=True)

        n_train_samples = len(x_train)
        # Identify and remove self-index per row
        self_mask = indices == np.arange(n_train_samples)[:, None]
        # Safety: if a row has no self (rare), fallback to dropping first neighbor
        if not np.all(np.sum(self_mask, axis=1) == 1):
            # Fallback: assume first column is self if missing
            enforced_mask = np.zeros_like(self_mask, dtype=bool)
            enforced_mask[np.arange(n_train_samples), 0] = True
            self_mask = np.where(
                np.sum(self_mask, axis=1, keepdims=True) == 1, self_mask, enforced_mask
            )
        keep_mask = ~self_mask
        distances_excl = distances[keep_mask].reshape(n_train_samples, max_k)
        indices_excl = indices[keep_mask].reshape(n_train_samples, max_k)

        y_array = y_train.to_numpy()
        neighbor_targets = y_array[indices_excl]  # shape (n, max_k)

        k_values = list(range(min_k, max_k + 1))
        scores: list[float] = []

        if self.weights == "uniform":
            # Precompute cumulative sums for O(1) mean retrieval per k
            cumsum_targets = np.cumsum(neighbor_targets, axis=1)
            for k in k_values:
                preds = cumsum_targets[:, k - 1] / k
                mse = self._calculate_mse(y_train, preds)
                adjusted_score = mse * ((k + 1) ** 2 / (k**2))
                scores.append(adjusted_score)
        else:  # distance weighting
            # For each k slice distances & targets then compute inverse-distance weighted average
            for k in k_values:
                d_k = distances_excl[:, :k]
                t_k = neighbor_targets[:, :k]
                # Zero distance (or numerically ~0) handling
                zero_mask_array = np.isclose(d_k, 0.0)
                any_zero = np.any(zero_mask_array, axis=1)
                preds = np.empty(n_train_samples, dtype=float)

                for i in range(n_train_samples):
                    row_targets = t_k[i, :]
                    if any_zero[i]:
                        # Average only zero-distance (or near-zero) targets
                        zero_targets = row_targets[zero_mask_array[i, :]]
                        preds[i] = float(np.mean(zero_targets))
                    else:
                        row_distances = d_k[i, :]
                        safe_distances = np.maximum(row_distances, np.finfo(float).eps)
                        weights_array = 1.0 / safe_distances
                        numerator = float(np.sum(weights_array * row_targets))
                        denominator = float(np.sum(weights_array))
                        # Fallback for extreme numerical stability issues.
                        if (
                            not np.isfinite(denominator)
                            or denominator < np.finfo(float).tiny
                        ):
                            preds[i] = float(np.mean(row_targets))
                        else:
                            preds[i] = numerator / denominator

                mse = self._calculate_mse(y_train, preds)
                adjusted_score = mse * ((k + 1) ** 2 / (k**2))
                scores.append(adjusted_score)

        if not scores or all(np.isinf(scores)):
            return min_k
        return k_values[np.argmin(scores)]

    def find_optimal_k_kfold(self, train_sets: MLSets) -> int:
        """Find optimal k via K-fold cross-validation MSE.

        Parameters
        ----------
        train_sets : MLSets
            Training features and target values.

        Returns
        -------
        int
            k minimizing mean CV MSE (>=1). Falls back to penalized LOO when
            sample count < number of folds.
        """
        n_samples = len(self.ml_data.df)
        x_train, y_train = train_sets.x, train_sets.y
        x_scaled = self._optimal_k_scaling(x_train)

        min_k, max_k = self._get_k_bounds(n_samples)
        if max_k <= min_k:
            return min_k

        # Ensure we have enough samples for k-fold CV
        if n_samples < self.optimal_k_cv_folds:
            # Fall back to leave-one-out if not enough samples for k-fold
            return self.find_optimal_k(train_sets)

        k_values = list(range(min_k, max_k + 1))
        cv_scores = []

        for k in k_values:
            fold_scores = []
            kfold = KFold(
                n_splits=self.optimal_k_cv_folds,
                shuffle=True,
                random_state=self.optimal_k_random_state,
            )

            for train_idx, test_idx in kfold.split(x_scaled):
                # Skip if training fold doesn't have enough samples for k neighbors
                if len(train_idx) < k:
                    fold_scores.append(np.inf)
                    continue

                knn = KNeighborsRegressor(
                    n_neighbors=k, algorithm=self.algorithm, weights=self.weights
                )
                knn.fit(x_scaled[train_idx], y_train.iloc[train_idx])
                y_pred = knn.predict(x_scaled[test_idx])
                mse = self._calculate_mse(y_train.iloc[test_idx], y_pred)
                fold_scores.append(mse)

            # Calculate mean CV score, handling infinite values
            finite_scores = [s for s in fold_scores if np.isfinite(s)]
            if finite_scores:
                cv_scores.append(np.mean(finite_scores))
            else:
                cv_scores.append(np.inf)

        if not cv_scores or all(np.isinf(cv_scores)):
            return min_k

        return k_values[np.argmin(cv_scores)]

    def fit(self) -> None:
        """Fit k-NN imputer on training data.

        Determines optimal k if not provided, fits scaling, and trains regressor.
        Ensures k >= 2 for sampling strategy when possible.

        Raises
        ------
        ValueError
            If training data is empty or scaling fails.
        """
        nona_sets = self.ml_data.nona_sets()
        if self.optimal_k is None:
            if self.optimal_k_method == "kfold":
                self.optimal_k = self.find_optimal_k_kfold(nona_sets)
            else:
                self.optimal_k = self.find_optimal_k(nona_sets)
        if self.optimal_k is None:
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
        """Execute imputation on missing target values.

        Parameters
        ----------
        random_state : int, optional
            Seed for reproducible sampling when strategy='sample'.

        Returns
        -------
        pd.DataFrame
            Complete dataframe with imputed target values.
        """
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
