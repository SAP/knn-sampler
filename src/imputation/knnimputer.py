import pandas as pd
from sklearn.impute import KNNImputer

from src.data_preparation.data_description import DataFrameMLData
from src.imputation import Imputer


class KnnImputer(Imputer):
    def __init__(self, ml_data: DataFrameMLData, n_neighbors: int = 5):
        super().__init__(ml_data=ml_data)
        # Initialize the knn_imputer attribute
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    def _execute(self) -> pd.DataFrame:
        imputed_df = self.ml_data.df
        # Impute missing values using KNNImputer
        imputed_values = self.knn_imputer.fit_transform(
            imputed_df[
                [
                    self.ml_data.dataset_descriptor.input_column,
                    self.ml_data.dataset_descriptor.target_column,
                ]
            ].to_numpy()
        )
        imputed_df[self.ml_data.dataset_descriptor.target_column] = imputed_values[:, 1]

        return imputed_df

    def fit(self):
        pass
