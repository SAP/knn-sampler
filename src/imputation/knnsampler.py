from typing import Literal, Optional, overload

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from src.data_preparation.data_description import DataFrameMLData, MLSets
from src.imputation import UncertaintyImputer
from src.imputation.imputer import BoundsPerPercentile


class KnnSampler(UncertaintyImputer):
    @overload
    def __init__(
        self,
        ml_data: DataFrameMLData,
        compute_bounds: Literal[False],
        lower_percentiles: None = None,
        upper_percentiles: None = None,
        n_neighbors: Optional[int] = None,
    ): ...
    @overload
    def __init__(
        self,
        ml_data: DataFrameMLData,
        compute_bounds: Literal[True],
        lower_percentiles: list[float],
        upper_percentiles: list[float],
        n_neighbors: Optional[int] = None,
    ): ...

    def __init__(
        self,
        ml_data: DataFrameMLData,
        compute_bounds: bool,
        lower_percentiles: list[float] | None = None,
        upper_percentiles: list[float] | None = None,
        n_neighbors: Optional[int] = None,
    ):
        super().__init__(
            ml_data=ml_data,
            lower_percentile=lower_percentiles or [5],
            upper_percentile=upper_percentiles or [95],
            compute_bounds=compute_bounds,
        )
        self.optimal_k: int | None = n_neighbors
        self.knn: KNeighborsRegressor | None = None

    def find_optimal_k(self, train_sets: MLSets):
        # No NaN values for x & y
        x_train, y_train = train_sets.x, train_sets.y

        # Standardize the features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)

        # Sample size
        n = len(x_train)

        # Maximum value for k based on the square root heuristic
        max_k = int(np.sqrt(n))

        # Test various numbers of neighbors, excluding the maximum k
        k_values = range(1, max_k)
        adjusted_scores_k: list[float] = []
        for k in k_values:
            # Add the provided code here
            regressor_k_plus_1 = KNeighborsRegressor(
                n_neighbors=k + 1, algorithm="kd_tree"
            )
            regressor_k_plus_1.fit(x_scaled, y_train)
            y_pred_k_plus_1 = regressor_k_plus_1.predict(x_scaled)
            mse_k_plus_1 = (y_train - y_pred_k_plus_1) ** 2
            adjusted_score = float(np.mean(mse_k_plus_1) * ((k + 1) ** 2 / k**2))
            adjusted_scores_k.append(adjusted_score)

        return k_values[np.argmin(adjusted_scores_k)]

    def fit(self):
        nona_sets = self.ml_data.nona_sets()
        self.optimal_k = (
            self.optimal_k
            if self.optimal_k is not None
            else self.find_optimal_k(nona_sets)
        )

        self.knn = KNeighborsRegressor(n_neighbors=self.optimal_k)
        self.knn.fit(nona_sets.x, nona_sets.y)

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
        if self.knn is None:
            raise ValueError("fitting required before imputation")
        imputed_target_values = pd.Series(dtype="float64")

        bounds: BoundsPerPercentile = {}
        for lp, up in zip(self.lower_percentiles, self.upper_percentiles, strict=True):
            lower_bounds: list[float] = []
            upper_bounds: list[float] = []
            bounds[(lp, up)] = (lower_bounds, upper_bounds)
            for index, row in df[
                df[self.ml_data.dataset_descriptor.target_column].isna()
            ].iterrows():
                row_reshaped = pd.DataFrame(
                    [row[self.ml_data.dataset_descriptor.input_column]],
                    columns=[self.ml_data.dataset_descriptor.input_column],
                )
                _, ind = self.knn.kneighbors(row_reshaped)

                # Upper and lower bounds
                if self.compute_bounds:
                    neighborhood = y_train.iloc[ind[0]].to_numpy()
                    lower_bounds.append(np.percentile(neighborhood, lp))
                    upper_bounds.append(np.percentile(neighborhood, up))

                # Impute missing values by sampling randomly from the k nearest neighbors
                random_sample = np.random.choice(ind[0])
                imputed_value = y_train.iloc[random_sample]
                imputed_target_values.at[index] = (
                    imputed_value  # Store the imputed value in the data frame
                )

        return imputed_target_values, bounds
