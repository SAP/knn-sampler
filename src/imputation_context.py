from dataclasses import dataclass

import pandas as pd

from src.data_preparation.data_description import DataFrameMLData, DatasetDescriptor


@dataclass
class ImputationContext:
    full_data: pd.DataFrame
    context_data: pd.DataFrame
    actual_data: pd.DataFrame
    predicted_data: pd.DataFrame
    dataset_descriptor: DatasetDescriptor
    execution_time: float | None

    @staticmethod
    def create_imputation_context(
        original_data: DataFrameMLData,
        actual_values: pd.Series,
        predicted_values: pd.DataFrame,
        execution_time: float | None = None,
    ) -> "ImputationContext":
        """
        Takes the original data (dataframe including missing values),
        the actual values,
        and the predicted values
        generates a plotting context for a dataset
        """
        original_df = original_data.df
        dataset_descriptor = original_data.dataset_descriptor
        y = dataset_descriptor.target_column

        # create a mask of missing values
        missing_mask = original_df[y].isna()

        # merge original and actual values to get the full dataframe
        full_df = original_df.copy()
        full_df.loc[missing_mask, y] = actual_values

        # create a dataframe with only the non-NaN values
        context_only_df = original_df[~missing_mask].copy()

        # reindex predicted_values before using [missing_mask]
        predicted_values = predicted_values.reindex(original_df.index)
        # create a dataframe with only x and y values for predicted data
        predicted_only_df = predicted_values[missing_mask]

        # create a dataframe with only x and y values for actual values
        actual_values_only_df = full_df[missing_mask]

        if actual_values_only_df.shape != predicted_only_df.shape:
            raise ValueError(
                "Predicted data and actual values are not of the same shape"
            )

        full_len, _ = full_df.shape
        context_len, _ = context_only_df.shape
        if context_len + len([i for i in missing_mask if i]) != full_len:
            raise ValueError("mismatch between context values and full_df values")

        return ImputationContext(
            full_df,
            context_only_df,
            actual_values_only_df,
            predicted_only_df,
            original_data.dataset_descriptor,
            execution_time=execution_time,
        )
