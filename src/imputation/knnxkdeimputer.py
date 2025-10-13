import numpy as np
import pandas as pd

from src.data_preparation.data_description import DataFrameMLData
from src.imputation import Imputer
from src.imputation.knnxkde import KNNxKDE, KNNxKdeMetric
from src.imputation.normalization import normalization, renormalization


class KNNxKDEImputer(Imputer):
    """KNN x KDE imputer for missing value imputation.

    Uses k-nearest neighbors with kernel density estimation to impute missing values.
    Data is normalized to [0, 1] using min-max normalization as per the paper implementation.

    Args:
        ml_data: DataFrameMLData containing the dataset
        h: KDE bandwidth parameter (default: 0.03)
        tau: Softmax temperature for neighbor weighting (default: 0.02)
        metric: Distance metric ("nan_eucl" or "nan_std_eucl")
    """

    def __init__(
        self,
        ml_data: DataFrameMLData,
        h=0.03,
        tau=1.0 / 50.0,
        metric: KNNxKdeMetric = "nan_std_eucl",
    ):
        super().__init__(ml_data)
        self.descriptor = ml_data.dataset_descriptor
        self.knnxkde = KNNxKDE(h=h, tau=tau, metric=metric)

    def fit(self):
        pass

    def _execute(self) -> pd.DataFrame:
        imputed_df = self.ml_data.df

        # Extract data as numpy array
        data_array = imputed_df[
            [
                self.ml_data.dataset_descriptor.input_column,
                self.ml_data.dataset_descriptor.target_column,
            ]
        ].to_numpy()

        # Normalize using paper's implementation
        norm_data, norm_params = normalization(data_array)

        # Run KNNxKDE imputation on normalized data
        imputed_samples = self.knnxkde.impute_samples(norm_data)
        if imputed_samples is None:
            raise ValueError("samples were not imputed")

        # Apply imputation
        for (idx, col_idx), samples in imputed_samples.items():
            norm_data[idx, col_idx] = np.random.choice(samples)

        # Renormalize back to original scale
        denorm_data = renormalization(norm_data, norm_params)

        # Update dataframe
        imputed_df[
            [
                self.ml_data.dataset_descriptor.input_column,
                self.ml_data.dataset_descriptor.target_column,
            ]
        ] = denorm_data

        return imputed_df
