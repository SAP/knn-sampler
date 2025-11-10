import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_preparation.data_description import DataFrameMLData
from src.imputation import Imputer
from src.imputation.knnxkde import KNNxKDE


class KNNxKDEImputer(Imputer):
    """Impute missing target values using kNNxKDE sampling.

    Structure aligned with other Imputer implementations:
    - Constructor stores model/tunables.
    - fit() kept as a no-op for interface consistency.
    - _execute() performs the actual imputation and returns a full DataFrame copy.

    compatibility_mode controls the data matrix structure passed to KNNxKDE:
    - True: [Index, X, Y] matrix (3 columns) - mimics original external workflows
    - False: [X, Y] matrix (2 columns) - clean implementation

    The indexing is required since the imputation algorithm relies on row/column indices,
    but it doesn't change the DataFrame structure itself or the logic.

    The extra index column changes distance calculations and neighbor selection,
    affecting imputation quality. Use True for backward compatibility with
    existing results, False for cleaner algorithmic behavior.
    """

    def __init__(
        self,
        ml_data: DataFrameMLData,
        h: float = 0.03,
        tau: float = 1.0 / 50.0,
        metric: str = "nan_std_eucl",
        compatibility_mode: bool = True,  # Mimics original workflow with index column
    ) -> None:
        super().__init__(ml_data=ml_data)
        self.descriptor = ml_data.dataset_descriptor
        self.knnxkde = KNNxKDE(h=h, tau=tau, metric=metric)
        self.compatibility_mode = compatibility_mode

    def fit(self) -> None:  # kept for API symmetry, no training required
        return None

    def _execute(self) -> pd.DataFrame:
        df = self.ml_data.df
        input_col = self.descriptor.input_column
        target_col = self.descriptor.target_column

        # avoid rewriting original data
        imputed_df = df.copy()
        scaler = MinMaxScaler()

        # Normalize only (X, Y)
        imputed_df[[input_col, target_col]] = scaler.fit_transform(
            imputed_df[[input_col, target_col]]
        )

        if self.compatibility_mode:
            # Add sequential index as first column - this changes the algorithmic behavior
            # by adding an extra dimension to the distance calculations
            index_values = np.arange(len(imputed_df)).reshape(-1, 1)
            data_matrix = np.column_stack(
                [
                    index_values,
                    imputed_df[input_col].values,
                    imputed_df[target_col].values,
                ]
            )
            samples = self.knnxkde.impute_samples(data_matrix)

            if samples is None or len(samples) == 0:
                # Denormalize and return original data
                imputed_df[[input_col, target_col]] = scaler.inverse_transform(
                    imputed_df[[input_col, target_col]]
                )
                return imputed_df

            # Apply imputed values - target is now in column 2 (due to index column)
            target_col_idx = 2
            for (row_idx, col_idx), draws in samples.items():
                if col_idx == target_col_idx and len(draws) > 0:
                    imputed_df.loc[row_idx, target_col] = np.random.choice(draws)
        else:
            # Standard mode: use only input and target columns as 2D matrix
            data_matrix = np.column_stack(
                [imputed_df[input_col].values, imputed_df[target_col].values]
            )
            samples = self.knnxkde.impute_samples(data_matrix)

            if samples is None or len(samples) == 0:
                # Denormalize and return original data
                imputed_df[[input_col, target_col]] = scaler.inverse_transform(
                    imputed_df[[input_col, target_col]]
                )
                return imputed_df

            # Apply imputed values - target is in column 1
            target_col_idx = 1
            for (row_idx, col_idx), draws in samples.items():
                if col_idx == target_col_idx and len(draws) > 0:
                    imputed_df.loc[row_idx, target_col] = np.random.choice(draws)

        # Denormalize (X, Y)
        imputed_df[[input_col, target_col]] = scaler.inverse_transform(
            imputed_df[[input_col, target_col]]
        )
        return imputed_df
