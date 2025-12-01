import pandas as pd
from sklearn.impute import KNNImputer

from src.data_preparation.data_description import DataFrameMLData
from src.imputation import Imputer


class KnnImputer(Imputer):
    """KNN imputer using scikit-learn's KNNImputer.

    Imputes missing target values jointly with input features using Euclidean
    distance over both columns. Deterministic implementation.

    Parameters
    ----------
    ml_data : DataFrameMLData
        Prepared ML data wrapper.
    n_neighbors : int, default=5
        Number of neighbors for imputation.

    Notes
    -----
    Deterministic imputer that ignores execute(random_state=...).
    Returns full DataFrame with imputed target values.
    """

    def __init__(self, ml_data: DataFrameMLData, n_neighbors: int = 5):
        super().__init__(ml_data=ml_data)
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self) -> None:
        """Fit KNN imputer - no pre-fitting required for this implementation."""
        pass

    def _execute(self, random_state: int | None = None) -> pd.DataFrame:
        """Execute KNN imputation on input and target columns.

        Parameters
        ----------
        random_state : int | None
            Ignored for this deterministic imputer.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values imputed via KNN algorithm.
        """
        df = self.ml_data.df
        input_col = self.ml_data.dataset_descriptor.input_column
        target_col = self.ml_data.dataset_descriptor.target_column

        # Impute missing values using KNNImputer
        imputed_values = self.knn_imputer.fit_transform(
            df[[input_col, target_col]].to_numpy()
        )
        df[input_col] = imputed_values[:, 0]
        df[target_col] = imputed_values[:, 1]

        return df
