import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor

from src.data_preparation.data_description import DataFrameMLData
from src.imputation.imputer import Imputer


class MICEImputer(Imputer):
    def __init__(
        self,
        ml_data: DataFrameMLData,
        n_neighbors: int = 5,
        max_iter: int = 20,
        random_state: int = 42,
    ):
        super().__init__(ml_data=ml_data)
        knn_imputer = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.mice_imputer = IterativeImputer(
            estimator=knn_imputer,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self):
        self.mice_imputer.fit(self.ml_data.df)

    def _execute(self) -> pd.DataFrame:
        imputed_df = self.ml_data.df
        imputed_values = self.mice_imputer.transform(
            imputed_df[
                [
                    self.ml_data.dataset_descriptor.input_column,
                    self.ml_data.dataset_descriptor.target_column,
                ]
            ]
        )
        imputed_df[self.ml_data.dataset_descriptor.input_column] = imputed_values[:, 0]
        imputed_df[self.ml_data.dataset_descriptor.target_column] = imputed_values[:, 1]
        return imputed_df
