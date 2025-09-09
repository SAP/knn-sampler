import itertools
import math
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.data_preparation.data_preparation import (
    DataGenerator,
    Mar,
    Mcar,
    MissingDataGenerator,
    Rate,
)
from src.imputation.knnsampler import KnnSampler
from src.imputation_context import ImputationContext

type Geometry = Literal["linear", "ring"]

lower_percentiles = [20.0, 10.0, 5.0]
upper_percentiles = [80.0, 90.0, 95.0]
sample_sizes = [1000, 2000, 10000]
missing_rates = [0.1, 0.3, 0.5]
geometries: list[Geometry] = ["linear", "ring"]
mechanisms: list[partial[MissingDataGenerator]] = [
    partial(Mcar),  # type: ignore
    partial(Mar, 0.5, 1.5),  # type: ignore
]
iterations = 2

type IterativeMetric = tuple[Geometry, partial[MissingDataGenerator], int, float, int]
type SingleResult = tuple[Geometry, MissingDataGenerator, float, int, float, int, float]
iterative_metrics: tuple[
    list[Geometry], list[partial[MissingDataGenerator]], list[int], list[float], range
] = (
    geometries,
    mechanisms,
    sample_sizes,
    missing_rates,
    range(iterations),
)


def plot_all_in_one(
    all_data: list[
        tuple[Geometry, MissingDataGenerator, list[pd.DataFrame], list[pd.DataFrame]]
    ],
    coverage_labels,
):
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
            "font.size": 12,
        }
    )

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 16))
    offset_unit = 0.05
    spacing_factor = 0.2

    all_handles = []
    all_labels = []

    for row_i, (geometry, mechanism, means_list, stds_list) in enumerate(all_data):
        for col_j in range(len(upper_percentiles)):
            ax = axes[row_i, col_j]

            mean_df = means_list[col_j]
            std_df = stds_list[col_j]
            ci_label = coverage_labels[col_j]

            sample_sizes = sorted(mean_df["Sample Size"].unique())
            missing_rates = sorted(mean_df["Missing Rate"].unique())

            x_base = np.arange(len(sample_sizes)) * spacing_factor

            for i_mr, mr in enumerate(missing_rates):
                subset_mean = mean_df[mean_df["Missing Rate"] == mr].sort_values(
                    "Sample Size"
                )
                subset_std = std_df[std_df["Missing Rate"] == mr].sort_values(
                    "Sample Size"
                )

                offset = (i_mr - (len(missing_rates) - 1) / 2.0) * offset_unit

                cp_vals = subset_mean["Coverage Probability"].values
                cp_errs = subset_std["Coverage Probability"].values

                if len(x_base) == len(cp_vals):
                    ax.errorbar(
                        x_base + offset,
                        cp_vals,
                        yerr=cp_errs,
                        marker="o",
                        markersize=7,
                        markeredgewidth=1.5,
                        elinewidth=2,
                        capsize=4,
                        linestyle="none",
                        label=f"MR={mr}",
                    )

            ax.set_title(f"{geometry} - {mechanism} - {ci_label} CI")
            ax.grid(alpha=0.4, linestyle="--")

            if col_j == 0:
                ax.set_ylabel("Coverage Probability")

            if row_i < 3:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Sample Size")
                ax.set_xticks(x_base)
                ax.set_xticklabels(sample_sizes, rotation=0)

            handle, label = ax.get_legend_handles_labels()
            all_handles.extend(handle)
            all_labels.extend(label)

    unique_handles_labels = dict(zip(all_labels, all_handles, strict=False))
    unique_labels = list(unique_handles_labels.keys())
    unique_handles = list(unique_handles_labels.values())

    fig.legend(unique_handles, unique_labels, loc="upper center", ncol=3, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.show()


type CoverageProbabilityKey = tuple[Geometry, MissingDataGenerator]


def process_combination(args: IterativeMetric):
    geometry, mech, size, rate, iteration = args
    mech = mech(missing_values=Rate(rate))  # type: ignore
    generator = DataGenerator(1, size, mech, geometry)  # type: ignore
    generated_data = generator.prepare_data()
    sampler = KnnSampler(
        generated_data,
        lower_percentiles=lower_percentiles,
        upper_percentiles=upper_percentiles,
        compute_bounds=True,
    )
    sampler.fit()
    results = sampler.execute()
    context = ImputationContext.create_imputation_context(
        generated_data, generator.actual_values, results
    )

    results_batch: list[
        tuple[Geometry, MissingDataGenerator, float, int, float, int, float]
    ] = []
    for lp, up in zip(lower_percentiles, upper_percentiles, strict=True):
        lb, ub = sampler.bounds[(lp, up)]
        coverage_probability = np.mean(
            (context.actual_data[generator.dataset_descriptor.target_column] >= lb)
            & (context.actual_data[generator.dataset_descriptor.target_column] <= ub)
        )
        results_batch.append(
            (geometry, mech, up, size, rate, iteration, float(coverage_probability))  # type: ignore
        )

    return results_batch


def transform_results(results: list[SingleResult]):
    result: dict[
        CoverageProbabilityKey, tuple[list[pd.DataFrame], list[pd.DataFrame]]
    ] = defaultdict(lambda: ([], []))
    groupedByGeoAndMissing: dict[
        CoverageProbabilityKey, list[tuple[float, int, float, int, float]]
    ] = defaultdict(list)

    for geometry, missing_type, up, size, rate, it, cp in results:
        groupedByGeoAndMissing[(geometry, missing_type)].append(
            (up, size, rate, it, cp)
        )

    for (geometry, missing_type), metrics in groupedByGeoAndMissing.items():
        mean_list, std_list = result[(geometry, missing_type)]
        for up in upper_percentiles:
            relevant_metrics = [m[1:] for m in metrics if m[0] == up]
            df = pd.DataFrame(
                data=relevant_metrics,
                columns=[
                    "Sample Size",
                    "Missing Rate",
                    "Iteration",
                    "Coverage Probability",
                ],
            )
            mean_df = (
                df.groupby(["Sample Size", "Missing Rate"])["Coverage Probability"]
                .mean()
                .reset_index()
            )
            std_df = (
                df.groupby(["Sample Size", "Missing Rate"])["Coverage Probability"]
                .std()
                .reset_index()
            )
            mean_list.append(mean_df)
            std_list.append(std_df)

    return [(*k, *v) for k, v in result.items()]


def main():
    combinations = list(product(*iterative_metrics))
    total_combinations = math.prod(map(len, iterative_metrics))  # type: ignore
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_combination, combinations),
                total=total_combinations,
            )
        )

    return results


if __name__ == "__main__":
    mp.freeze_support()
    results = main()

    res: list[SingleResult] = [*itertools.chain(*results)]
    plot_all_in_one(
        transform_results(res),
        coverage_labels=[
            f"{lp}-{up}%"
            for lp, up in zip(lower_percentiles, upper_percentiles, strict=True)
        ],
    )
