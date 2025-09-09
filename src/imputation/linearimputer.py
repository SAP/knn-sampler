import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_preparation.data_description import DataFrameMLData
from src.imputation.imputer import Imputer


class LinearImputer(Imputer):
    def __init__(self, ml_data: DataFrameMLData):
        super().__init__(ml_data=ml_data)
        self.linear_model = LinearRegression()

    def fit(self) -> None:
        nona_sets = self.ml_data.nona_sets()
        self.linear_model.fit(nona_sets.x, nona_sets.y)

    def _execute(self) -> pd.DataFrame:
        df = self.ml_data.df
        input_col = self.ml_data.dataset_descriptor.input_column
        target_col = self.ml_data.dataset_descriptor.target_column

        mask = df[target_col].isna() & df[input_col].notna()
        if not mask.any():
            return df.loc[mask].copy()

        imputed_df = df.loc[mask, [input_col, target_col]].copy()
        imputed_df[target_col] = self.linear_model.predict(imputed_df[[input_col]])
        return imputed_df
