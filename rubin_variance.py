import numpy as np
import pandas as pd

from src.data_preparation.data_preparation import DataGenerator, Mcar, Rate
from src.imputation.knnsampler import KnnSampler
from src.utils import do_imputation


def rubin_variance(
    imputed_dfs: list[pd.DataFrame],
) -> tuple[np.floating, float, np.floating]:
    m = len(imputed_dfs)
    # Calculate the mean and variance for each dataset
    means = np.zeros(m)
    variances = np.zeros(m)
    for i, imputed_df in enumerate(imputed_dfs):
        means[i] = imputed_df.Y.mean()
        variances[i] = imputed_df["Y"].var(ddof=1)  # type: ignore

    # Within-imputation variance (mean of within-imputation variances)
    W = np.mean(variances)

    # Between-imputation variance
    B = (1 / (m - 1)) * np.sum((means - np.mean(means)) ** 2)

    # Total variance
    T = W + ((1 + 1 / m) * B)

    return W, B, T


generator = DataGenerator(
    linear_interpolation_ratio=1,
    sample_size=100,
    missing_generator=Mcar(Rate(0.1)),
    geometry_type="linear",
)

data = generator.prepare_data()

sampler = KnnSampler(
    ml_data=data, lower_percentiles=[5], upper_percentiles=[95], compute_bounds=True
)

imputation_results = do_imputation(sampler, iterations=5)[sampler]

resulted_dataframes = [df for df, _ in imputation_results]

w, b, total_variance = rubin_variance(resulted_dataframes)

print("Within-imputation variance: ", w)
print("Between-imputation variance: ", b)

print("total variance : ", total_variance)


for i, imputed_df in enumerate(resulted_dataframes):
    print(f"Description for Imputation {i + 1}:")
    print(imputed_df.describe())

    mode = imputed_df.mode().iloc[0]

    print("mode:\n", mode)

    print("median", imputed_df.median())
