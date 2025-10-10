import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_preparation.data_description import DataFrameMLData
from src.imputation import Imputer
from src.imputation.knnxkde import KNNxKDE, KNNxKdeMetric


class KNNxKDEImputer(Imputer):
    """Hybrid KNN + KDE sampling imputer for missing target values.

    Applies MinMax scaling, finds KNN neighborhoods, generates candidate samples
    via kernel density estimation over neighbor values, then randomly samples
    one value per missing target.

    Parameters
    ----------
    ml_data : DataFrameMLData
        Prepared ML data wrapper.
    h : float, default=0.03
        KDE bandwidth parameter for kernel density estimation.
    tau : float, default=0.02
        Temperature parameter for distance-based neighbor weighting.
    metric : {'nan_eucl', 'nan_std_eucl'}, default='nan_std_eucl'
        Distance metric for KNNxKDE backend:
        - 'nan_eucl': NaN-aware Euclidean distance
        - 'nan_std_eucl': NaN-aware standardized Euclidean distance

    Notes
    -----
    Stochastic imputer using global NumPy RNG.
    Currently ignores execute(random_state=...) - could be extended for reproducibility.
    Returns full DataFrame with imputed target values.
    """

    def __init__(
        self,
        ml_data: DataFrameMLData,
        h=0.03,
        tau=50.0 / 1000.0,
        metric: KNNxKdeMetric = "nan_std_eucl",
    ):
        super().__init__(ml_data)
        self.descriptor = ml_data.dataset_descriptor
        self.knnxkde = KNNxKDE(h=h, tau=tau, metric=metric)

    def fit(self):
        """Fit KNNxKDE imputer - no pre-fitting required for this implementation."""
        pass

    def _execute(self, random_state: int | None = None) -> pd.DataFrame:
        """Execute KNNxKDE imputation with scaling and sampling.

        Parameters
        ----------
        random_state : int | None
            Currently ignored - uses global NumPy RNG for sampling.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing target values imputed via KNNxKDE sampling.

        Raises
        ------
        ValueError
            If imputation samples cannot be generated.
        """
        df = self.ml_data.df
        input_col = self.ml_data.dataset_descriptor.input_column
        target_col = self.ml_data.dataset_descriptor.target_column

        scaler = MinMaxScaler()
        df[[input_col, target_col]] = scaler.fit_transform(df[[input_col, target_col]])

        imputed_samples = self.knnxkde.impute_samples(df.to_numpy())
        if imputed_samples is None:
            raise ValueError("KNNxKDE imputation failed: no suitable samples generated")

        for (idx, col_idx), samples in imputed_samples.items():
            if col_idx == 1:  # target column index after scaling
                df.loc[idx, target_col] = np.random.choice(samples)

        # Inverse transform to original scale
        df[[input_col, target_col]] = scaler.inverse_transform(
            df[[input_col, target_col]]
        )

        return df
