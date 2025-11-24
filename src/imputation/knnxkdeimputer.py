import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_preparation.data_description import DataFrameMLData
from src.imputation import Imputer
from src.imputation.knnxkde import KNNxKDE, KNNxKdeMetric


class KNNxKDEImputer(Imputer):
    """Impute missing target values using kNNxKDE sampling.

    The implementation adds a sequential index column to the data matrix, resulting in a matrix with the following column order:
        - Column 0: Sequential index
        - Column 1: Input feature (X)
        - Column 2: Target feature (Y)

    This ordering is critical: TARGET_COL_IDX = 2 refers to the target column in the [Index, X, Y] matrix.
    The index column participates in distance calculations and affects neighbor selection, preserving the original kNNxKDE algorithm behavior from the research paper.
    """

    TARGET_COL_IDX = 2  # Index of target column in [Index, X, Y] matrix

    def __init__(
        self,
        ml_data: DataFrameMLData,
        h: float = 0.03,
        tau: float = 1.0 / 50.0,
        metric: KNNxKdeMetric = "nan_std_eucl",
    ) -> None:
        super().__init__(ml_data=ml_data)
        self.descriptor = ml_data.dataset_descriptor
        self.knnxkde = KNNxKDE(h=h, tau=tau, metric=metric)

    def fit(self) -> None:
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
        input_col = self.descriptor.input_column
        target_col = self.descriptor.target_column

        scaler = MinMaxScaler()
        df[[input_col, target_col]] = scaler.fit_transform(df[[input_col, target_col]])

        index_values = np.arange(len(df)).reshape(-1, 1)
        data_matrix = np.column_stack(
            [
                index_values,
                df[input_col].values,
                df[target_col].values,
            ]
        )
        samples = self.knnxkde.impute_samples(data_matrix)

        if samples is None or len(samples) == 0:
            df[[input_col, target_col]] = scaler.inverse_transform(
                df[[input_col, target_col]]
            )
            return df

        for (row_idx, col_idx), draws in samples.items():
            if col_idx == self.TARGET_COL_IDX and len(draws) > 0:
                df.loc[row_idx, target_col] = np.random.choice(draws)

        df[[input_col, target_col]] = scaler.inverse_transform(
            df[[input_col, target_col]]
        )

        return df
