import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data_preparation.data_description import DataFrameMLData
from src.imputation.imputer import Imputer


class RandomForestImputer(Imputer):
    """Random forest regression imputer for missing target values.

    Trains RandomForestRegressor on complete cases and predicts missing
    target values where input features are available.

    Parameters
    ----------
    ml_data : DataFrameMLData
        Prepared ML data wrapper.
    random_state : int, default=42
        Seed for RandomForestRegressor reproducibility.

    Notes
    -----
    Stochastic during model training, deterministic during prediction.
    Ignores execute(random_state=...) - uses constructor random_state only.
    Returns full DataFrame with imputed target values.
    """

    def __init__(self, ml_data: DataFrameMLData, random_state: int = 42):
        super().__init__(ml_data=ml_data)
        self.rf = RandomForestRegressor(random_state=random_state)

    def fit(self):
        """Fit random forest model on complete cases."""
        sets = self.ml_data.nona_sets()
        self.rf.fit(sets.x, sets.y)

    def _execute(self, random_state: int | None = None) -> pd.DataFrame:
        """Execute imputation using fitted random forest model.

        Parameters
        ----------
        random_state : int | None
            Ignored - uses constructor random_state for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing target values imputed via random forest regression.
        """
        df = self.ml_data.df
        target_col = self.ml_data.dataset_descriptor.target_column
        input_col = self.ml_data.dataset_descriptor.input_column

        mask = df[target_col].isna() & df[input_col].notna()
        if mask.any():
            df.loc[mask, target_col] = self.rf.predict(df.loc[mask, [input_col]])

        return df
