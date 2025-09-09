import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_preparation.data_description import DataFrameMLData
from src.imputation import Imputer
from src.imputation.knnxkde import KNNxKDE, KNNxKdeMetric


class KNNxKDEImputer(Imputer):
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
        pass

    def _execute(self) -> pd.DataFrame:
        imputed_df = self.ml_data.df
        scaler = MinMaxScaler()
        imputed_df[
            [
                self.ml_data.dataset_descriptor.input_column,
                self.ml_data.dataset_descriptor.target_column,
            ]
        ] = scaler.fit_transform(
            imputed_df[
                [
                    self.ml_data.dataset_descriptor.input_column,
                    self.ml_data.dataset_descriptor.target_column,
                ]
            ]
        )
        imputed_samples = self.knnxkde.impute_samples(imputed_df.to_numpy())
        if imputed_samples is None:
            raise ValueError("samples were not imputed")
        for (idx, _), samples in imputed_samples.items():
            imputed_df.loc[idx, self.ml_data.dataset_descriptor.target_column] = (
                np.random.choice(samples)
            )
        imputed_df[
            [
                self.ml_data.dataset_descriptor.input_column,
                self.ml_data.dataset_descriptor.target_column,
            ]
        ] = scaler.inverse_transform(
            imputed_df[
                [
                    self.ml_data.dataset_descriptor.input_column,
                    self.ml_data.dataset_descriptor.target_column,
                ]
            ]
        )
        return imputed_df
