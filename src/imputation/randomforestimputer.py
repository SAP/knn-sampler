import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data_preparation.data_description import DataFrameMLData
from src.imputation.imputer import Imputer


class RandomForestImputer(Imputer):
    def __init__(self, ml_data: DataFrameMLData, random_state: int = 42):
        super().__init__(ml_data=ml_data)
        self.ml_data: DataFrameMLData
        self.rf = RandomForestRegressor(random_state=random_state)

    def fit(self):
        sets = self.ml_data.nona_sets()
        self.rf.fit(sets.x, sets.y)

    def _execute(self) -> pd.DataFrame:
        df = self.ml_data.df
        imputed_df = df[df[self.ml_data.dataset_descriptor.target_column].isna()]
        imputed_df.loc[:, self.ml_data.dataset_descriptor.target_column] = (
            self.rf.predict(imputed_df[[self.ml_data.dataset_descriptor.input_column]])
        )
        return imputed_df
