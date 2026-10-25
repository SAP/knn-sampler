from typing import Literal, cast, overload

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data_preparation.data_description import DataFrameMLData, MLSets
from src.imputation import UncertaintyImputer
from src.imputation.imputer import BoundsPerPercentile, validate_random_state

ImputationStrategy = Literal["sample", "mean", "median"]
KNNAlgorithm = Literal["auto", "ball_tree", "kd_tree", "brute"]
ScalingStrategy = Literal["none", "standardization", "normalization"]
KOptimalMethod = Literal["loo_penalized", "kfold"]
KNNWeights = Literal["uniform", "distance"]

IMPUTATION_STRATEGIES: tuple[ImputationStrategy, ...] = ("sample", "mean", "median")
KNN_ALGORITHMS: tuple[KNNAlgorithm, ...] = ("auto", "ball_tree", "kd_tree", "brute")
SCALING_STRATEGIES: tuple[ScalingStrategy, ...] = (
    "none",
    "standardization",
    "normalization",
)
K_OPTIMAL_METHODS: tuple[KOptimalMethod, ...] = ("loo_penalized", "kfold")
KNN_WEIGHTS: tuple[KNNWeights, ...] = ("uniform", "distance")

DEFAULT_LOWER_PERCENTILES: list[float] = [5.0]
DEFAULT_UPPER_PERCENTILES: list[float] = [95.0]
DEFAULT_RANDOM_STATE: int = 42
MAX_KFOLD_RANDOM_STATE: int = 2**32 - 1
DEFAULT_CV_FOLDS: int = 3


def _validate_kfold_random_state(random_state: int | None) -> None:
    """Validate and normalize a K-fold seed and check its range."""
    random_state = validate_random_state(random_state, option="optimal_k_random_state")
    if random_state is not None and random_state > MAX_KFOLD_RANDOM_STATE:
        raise ValueError(
            f"optimal_k_random_state must be <= 2**32 - 1, got {random_state}"
        )


def validate_option(name: str, value: str, allowed: tuple[str, ...]) -> None:
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
        Seed for K-Fold splitting when optimal_k_method='kfold'; defaults to 42.
        With kfold, it must be in [0, 2**32 - 1]. Set to None for
        nondeterministic fold selection (Python API only; configuration files
        accept integers). Validated for every method, but unused by
        loo_penalized.
    optimal_k_cv_folds : int
        Number of CV folds (>=2) for optimal_k_method='kfold'.
    scaling_optimal_k : {'none', 'standardization', 'normalization'}
        Scaling applied during k selection phase. For kfold, the scaler is
        fitted independently on each training fold; loo_penalized keeps the
        historical global fit.
    scaling_fit : {'none', 'standardization', 'normalization'}
        Scaling applied for final fitting and imputation.

    Notes
    -----
    Execution randomness is controlled via execute(random_state=...). When
    executed through ``do_imputation`` with a run seed, each iteration receives
    an independent derived seed; without a run seed, sampled executions remain
    nondeterministic.
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
        n_neighbors: int | None = None,
        strategy: ImputationStrategy = "sample",
        algorithm: KNNAlgorithm = "kd_tree",
        weights: KNNWeights = "uniform",
        optimal_k_method: KOptimalMethod = "loo_penalized",
        optimal_k_random_state: int | None = DEFAULT_RANDOM_STATE,
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
        n_neighbors: int | None = None,
        strategy: ImputationStrategy = "sample",
        algorithm: KNNAlgorithm = "kd_tree",
        weights: KNNWeights = "uniform",
        optimal_k_method: KOptimalMethod = "loo_penalized",
        optimal_k_random_state: int | None = DEFAULT_RANDOM_STATE,
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
        n_neighbors: int | None = None,
        strategy: ImputationStrategy = "sample",
        algorithm: KNNAlgorithm = "kd_tree",
        weights: KNNWeights = "uniform",
        optimal_k_method: KOptimalMethod = "loo_penalized",
        optimal_k_random_state: int | None = DEFAULT_RANDOM_STATE,
        optimal_k_cv_folds: int = DEFAULT_CV_FOLDS,
        scaling_optimal_k: ScalingStrategy = "standardization",
        scaling_fit: ScalingStrategy = "none",
    ):
        # Defaults percentiles
        lower_percentiles = lower_percentiles or DEFAULT_LOWER_PERCENTILES
        upper_percentiles = upper_percentiles or DEFAULT_UPPER_PERCENTILES

        if n_neighbors is not None and n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")

        validate_option("strategy", strategy, IMPUTATION_STRATEGIES)
        validate_option("algorithm", algorithm, KNN_ALGORITHMS)
        for label, value in (
            ("scaling_optimal_k", scaling_optimal_k),
            ("scaling_fit", scaling_fit),
        ):
            validate_option(label, value, SCALING_STRATEGIES)
        validate_option("optimal_k_method", optimal_k_method, K_OPTIMAL_METHODS)
        validate_option("weights", weights, KNN_WEIGHTS)
        if optimal_k_cv_folds < 2:
            raise ValueError(
                f"optimal_k_cv_folds must be >= 2, got {optimal_k_cv_folds}"
            )
        optimal_k_random_state = validate_random_state(
            optimal_k_random_state, option="optimal_k_random_state"
        )
        if optimal_k_method == "kfold":
            _validate_kfold_random_state(optimal_k_random_state)

        super().__init__(
            ml_data=ml_data,
            lower_percentiles=lower_percentiles,
            upper_percentiles=upper_percentiles,
            compute_bounds=compute_bounds,
        )
        self.optimal_k: int | None = n_neighbors
        self.knn: KNeighborsRegressor | None = None
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.strategy: ImputationStrategy = strategy
        self.algorithm: KNNAlgorithm = algorithm
        self.weights: KNNWeights = weights
        self.optimal_k_method: KOptimalMethod = optimal_k_method
        self.optimal_k_random_state: int | None = optimal_k_random_state
        self.optimal_k_cv_folds: int = optimal_k_cv_folds
        self.scaling_optimal_k: ScalingStrategy = scaling_optimal_k
        self.scaling_fit: ScalingStrategy = scaling_fit

    def _build_scaler(
        self, strategy: ScalingStrategy
    ) -> StandardScaler | MinMaxScaler | None:
        """Build an unfitted scaler for a scaling strategy."""
        if strategy == "standardization":
            return StandardScaler()
        if strategy == "normalization":
            return MinMaxScaler()
        return None

    def _optimal_k_scaling(self, X: pd.DataFrame) -> np.ndarray:
        """Apply scaling transformation for k selection phase.

        The scaler is fitted on all rows in ``X``. This is the historical
        leave-one-out path; the K-fold path fits one scaler per training fold.

        Parameters
        ----------
        X : pd.DataFrame
            Input features to transform.

        Returns
        -------
        np.ndarray
            Scaled feature matrix according to scaling_optimal_k strategy.
        """
        scaler = self._build_scaler(self.scaling_optimal_k)
        if scaler is None:
            return X.to_numpy()
        return cast(np.ndarray, scaler.fit_transform(X))

    def _scale_optimal_k_fold(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit k-selection scaling on one training fold and transform both folds."""
        scaler = self._build_scaler(self.scaling_optimal_k)
        if scaler is None:
            return x_train.to_numpy(), x_test.to_numpy()
        return (
            cast(np.ndarray, scaler.fit_transform(x_train)),
            cast(np.ndarray, scaler.transform(x_test)),
        )

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

        scaler = self._build_scaler(self.scaling_fit)
        self.scaler = None if scaler is None else scaler.fit(X)

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
        return cast(np.ndarray, self.scaler.transform(X))

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that the descriptor input and target columns are present.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Raises
        ------
        ValueError
            If the input or target column is missing.
        """
        descriptor = self.ml_data.dataset_descriptor
        missing = [
            column
            for column in (descriptor.input_column, descriptor.target_column)
            if column not in df.columns
        ]
        if missing:
            formatted = ", ".join(repr(column) for column in missing)
            raise ValueError(
                f"Missing required column(s): {formatted}. Expected input column"
                f" '{descriptor.input_column}' and target column"
                f" '{descriptor.target_column}'"
            )

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
        n_samples: int
            Number of complete-case training rows available for k search.

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
        max_k = int(min(heuristic_max_k, structural_max_k))
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

        Notes
        -----
        The k-selection scaler is fitted on all complete-case rows, including
        the row being left out. This path is retained for publication
        compatibility; use optimal_k_method='kfold' for fold-local scaling.
        """
        n_samples = len(train_sets.x)
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
            the number of complete-case training rows is smaller than the
            number of folds.

        Raises
        ------
        ValueError
            If optimal_k_random_state is outside scikit-learn's accepted range.

        Notes
        -----
        Folds are materialized once and reused for every candidate k so that
        all candidates are scored on the same split. The k-selection scaler is
        fitted independently on each training fold and applied to its
        validation fold, so validation rows do not influence scaling. With
        ``optimal_k_random_state=None``, the split is therefore still
        nondeterministic across calls but consistent within one selection.
        """
        _validate_kfold_random_state(self.optimal_k_random_state)
        n_samples = len(train_sets.x)
        x_train, y_train = train_sets.x, train_sets.y

        min_k, max_k = self._get_k_bounds(n_samples)
        if max_k <= min_k:
            return min_k

        # Ensure we have enough samples for k-fold CV
        if n_samples < self.optimal_k_cv_folds:
            # Fall back to leave-one-out if not enough samples for k-fold
            return self.find_optimal_k(train_sets)

        k_values = list(range(min_k, max_k + 1))
        cv_scores: list[float] = []

        kfold = KFold(
            n_splits=self.optimal_k_cv_folds,
            shuffle=True,
            random_state=self.optimal_k_random_state,
        )
        folds: list[tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]] = []
        for train_idx, test_idx in kfold.split(x_train):
            x_fold_train, x_fold_test = self._scale_optimal_k_fold(
                x_train.iloc[train_idx], x_train.iloc[test_idx]
            )
            folds.append(
                (
                    x_fold_train,
                    x_fold_test,
                    y_train.iloc[train_idx],
                    y_train.iloc[test_idx],
                )
            )

        for k in k_values:
            fold_scores = []

            for x_fold_train, x_fold_test, y_fold_train, y_fold_test in folds:
                # Skip if training fold doesn't have enough samples for k neighbors
                if len(x_fold_train) < k:
                    fold_scores.append(np.inf)
                    continue

                knn = KNeighborsRegressor(
                    n_neighbors=k, algorithm=self.algorithm, weights=self.weights
                )
                knn.fit(x_fold_train, y_fold_train)
                y_pred = knn.predict(x_fold_test)
                mse = self._calculate_mse(y_fold_test, y_pred)
                fold_scores.append(mse)

            # Calculate mean CV score, handling infinite values
            finite_scores = [s for s in fold_scores if np.isfinite(s)]
            if finite_scores:
                cv_scores.append(float(np.mean(finite_scores)))
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
            If a required column is missing, training data is empty, or scaling
            fails.

        Notes
        -----
        Calling ``fit()`` again refits the scaler and kNN regressor on the
        current data. A provided ``n_neighbors`` and an already selected
        ``optimal_k`` are reused; set ``optimal_k`` to None before fitting to
        force reselection after the data changed.
        """
        self._validate_columns(self.ml_data.dataframe)
        nona_sets = self.ml_data.nona_sets()
        if nona_sets.x.empty:
            raise ValueError(
                "No complete-case training rows: cannot fit the imputer"
                " (check the missing rate or the dataset)"
            )
        if self.optimal_k is None:
            if self.optimal_k_method == "kfold":
                self.optimal_k = self.find_optimal_k_kfold(nona_sets)
            else:
                self.optimal_k = self.find_optimal_k(nona_sets)

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
        random_state : int | None
            Seed for reproducible sampling when strategy='sample'. It is
            validated for every strategy.

        Returns
        -------
        pd.DataFrame
            Complete dataframe with imputed target values.

        Raises
        ------
        ValueError
            If random_state is invalid, a required column is missing, or the
            model is not fitted.
        """
        self._validate_columns(self.ml_data.dataframe)
        imputed_df = self.ml_data.df
        imputed_target_values, self.bounds = self.impute_for_dataset(
            imputed_df, self.ml_data.nona_sets().y, random_state
        )
        imputed_df.loc[
            imputed_target_values.index, self.ml_data.dataset_descriptor.target_column
        ] = imputed_target_values
        return imputed_df

    def impute_for_dataset(
        self, df: pd.DataFrame, y_train: pd.Series, random_state: int | None = None
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

        Raises
        ------
        ValueError
            If random_state is invalid, the target column is missing, the model
            is not fitted, or the input column is missing on a row that needs
            target imputation.
        """
        if self.knn is None:
            raise ValueError("Model not fitted: call fit() before imputation")
        random_state = validate_random_state(random_state)
        imputed_series = pd.Series(dtype="float64")
        bounds: BoundsPerPercentile = {
            (lp, up): ([], [])
            for lp, up in zip(
                self.lower_percentiles, self.upper_percentiles, strict=True
            )
        }
        descriptor = self.ml_data.dataset_descriptor
        target_col = descriptor.target_column
        input_col = descriptor.input_column
        if target_col not in df.columns:
            # Validate before indexing so a missing target gets an actionable
            # message instead of a raw KeyError.
            self._validate_columns(df)
        missing_mask = df[target_col].isna()
        missing_rows = df.loc[missing_mask]
        if missing_rows.empty:
            return imputed_series, bounds
        # Validate the input column only when target values need imputation;
        # callers may inspect fully observed targets without providing inputs.
        self._validate_columns(df)
        missing_inputs = missing_rows[input_col].isna()
        if missing_inputs.any():
            indexes = missing_rows.index[missing_inputs].tolist()
            preview = ", ".join(str(index) for index in indexes[:5])
            if len(indexes) > 5:
                preview += ", ..."
            raise ValueError(
                f"Cannot impute missing target values: input column '{input_col}'"
                f" has missing values in {len(indexes)} row(s) that require"
                f" imputation (indices: {preview})"
            )
        X_missing = missing_rows[[input_col]]
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
            rng = np.random.default_rng(random_state)
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
