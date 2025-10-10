import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor

from src.data_preparation.data_description import DataFrameMLData
from src.imputation.imputer import Imputer


class MICEImputer(Imputer):
    """Multiple Imputation by Chained Equations using IterativeImputer.

    Uses KNeighborsRegressor within scikit-learn's IterativeImputer for
    iterative imputation of missing values in both input and target columns.

    Parameters
    ----------
    ml_data : DataFrameMLData
        Prepared ML data wrapper.
    n_neighbors : int, default=5
        Number of neighbors for KNN estimator.
    max_iter : int, default=20
        Maximum MICE iterations.
    random_state : int, default=42
        Seed for reproducible iterative imputation.

    Notes
    -----
    Deterministic given internal seed, ignores execute(random_state=...).
    Returns full DataFrame with both input and target columns potentially modified.
    """

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
        """Fit MICE imputer on the dataset."""
        self.mice_imputer.fit(self.ml_data.df)

    def _execute(self, random_state: int | None = None) -> pd.DataFrame:
        """Execute MICE imputation on input and target columns.

        Parameters
        ----------
        random_state : int | None
            Ignored - uses constructor random_state for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values imputed via MICE algorithm.
        """
        df = self.ml_data.df
        input_col = self.ml_data.dataset_descriptor.input_column
        target_col = self.ml_data.dataset_descriptor.target_column

        imputed_values = self.mice_imputer.transform(df[[input_col, target_col]])
        df[input_col] = imputed_values[:, 0]
        df[target_col] = imputed_values[:, 1]

        return df
