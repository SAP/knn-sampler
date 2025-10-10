import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_preparation.data_description import DataFrameMLData
from src.imputation.imputer import Imputer


class LinearImputer(Imputer):
    """Linear regression imputer for missing target values.

    Fits univariate linear regression (target ~ input) on complete cases,
    then predicts missing targets where input features are available.

    Parameters
    ----------
    ml_data : DataFrameMLData
        Prepared ML data wrapper.

    Notes
    -----
    Deterministic imputer that ignores execute(random_state=...).
    Returns full DataFrame with imputed target values, not just imputed rows.
    """

    def __init__(self, ml_data: DataFrameMLData):
        super().__init__(ml_data=ml_data)
        self.linear_model = LinearRegression()

    def fit(self) -> None:
        """Fit linear regression model on complete cases."""
        nona_sets = self.ml_data.nona_sets()
        self.linear_model.fit(nona_sets.x, nona_sets.y)

    def _execute(self, random_state: int | None = None) -> pd.DataFrame:
        """Execute imputation using fitted linear model.

        Parameters
        ----------
        random_state : int | None
            Ignored for this deterministic imputer.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing target values imputed via linear regression.
        """
        df = self.ml_data.df
        input_col = self.ml_data.dataset_descriptor.input_column
        target_col = self.ml_data.dataset_descriptor.target_column

        mask = df[target_col].isna() & df[input_col].notna()
        if mask.any():
            df.loc[mask, target_col] = self.linear_model.predict(
                df.loc[mask, [input_col]]
            )

        return df
