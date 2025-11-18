import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_preparation.data_description import DataFrameMLData
from src.imputation import Imputer
from src.imputation.knnxkde import KNNxKDE, KNNxKdeMetric


class KNNxKDEImputer(Imputer):
    """Impute missing target values using kNNxKDE sampling.

    Structure aligned with other Imputer implementations:
    - Constructor stores model/tunables.
    - fit() kept as a no-op for interface consistency.
    - _execute() performs the actual imputation and returns a full DataFrame copy.

    The implementation uses a data matrix structure [Index, X, Y] (3 columns) passed to KNNxKDE,
    mimicking the original external workflows. The indexing is required since the imputation
    algorithm relies on row/column indices, but it doesn't change the DataFrame structure itself
    or the logic.

    The extra index column changes distance calculations and neighbor selection,
    affecting imputation quality.
    """

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
        pass

    def _execute(self) -> pd.DataFrame:
        imputed_df = self.ml_data.df
        input_col = self.descriptor.input_column
        target_col = self.descriptor.target_column

        scaler = MinMaxScaler()
        imputed_df[[input_col, target_col]] = scaler.fit_transform(
            imputed_df[[input_col, target_col]]
        )

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
            imputed_df[[input_col, target_col]] = scaler.inverse_transform(
                imputed_df[[input_col, target_col]]
            )
            return imputed_df

        target_col_idx = 2
        for (row_idx, col_idx), draws in samples.items():
            if col_idx == target_col_idx and len(draws) > 0:
                imputed_df.loc[row_idx, target_col] = np.random.choice(draws)

        imputed_df[[input_col, target_col]] = scaler.inverse_transform(
            imputed_df[[input_col, target_col]]
        )
        return imputed_df
