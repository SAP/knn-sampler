import math
import time
import typing
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Literal, Mapping

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange

from assets.profiling.line_profile import Profiler
from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import (
    DataGenerator,
    Mar,
    Mcar,
    Rate,
)
from src.imputation import (
    KnnImputer,
    KnnSampler,
    KNNxKDEImputer,
    LinearImputer,
    RandomForestImputer,
)
from src.imputation.imputer import Imputer
from src.imputation_context import ImputationContext
from src.stats_utils import (
    calculate_p_value,
    multivariate_energy_distance,
    permutation_test,
)
from src.utils import do_imputation, instantiate_imputers


@dataclass
class MissingConfig:
    missing_generator: Mcar | Mar
    sample_sizes: list[int]
    n_permutations: int


# 30% for each size
mar_config = MissingConfig(Mar(0.5, 1.5, Rate(0.3)), [3000], 200)
mcar_config = MissingConfig(Mcar(Rate(0.3)), [3000, 5000], 200)

###### configuration selection ######
config: MissingConfig = mcar_config
benchmark_geometry_type: Literal["linear", "ring"] = "ring"
#####################################

imputer_classes: dict[type[Imputer], dict[str, typing.Any]] = {
    KnnSampler: {
        "upper_percentiles": [97.5],
        "lower_percentiles": [2.5],
        "compute_bounds": True,
    },
    KnnImputer: {"n_neighbors": 5},
    RandomForestImputer: {"random_state": 42},
    KNNxKDEImputer: {"h": 0.03, "tau": 0.05},
    LinearImputer: {},
}

## KnnSampler config ##
iterations = 5
#######################

# Data regeneration policy
# When True, regenerate a fresh synthetic dataset for each imputer and for each sample size (per method/per iteration),
# to mimic the behavior of the original script and reproduce higher variance across methods.
REGENERATE_DATA_PER_METHOD = True

# type definitions

# results for each imputer
type ImputersResults = Mapping[Imputer, float | np.floating]

# mean and std values (for each imputer name)
type Metric = tuple[dict[str, float], dict[str, float]]

# rmse, energy distance, p mean and std values (for each imputer name) and execution times for each imputer for each range
type Benchmark = tuple[Metric, Metric, Metric, list[dict[str, str]]]


# root_rmse_values: dict[str, list[np.floating | float]]
# root_ed_values: dict[str, list[np.floating | float]]
# root_p_values: dict[str, list[float]]
# root_et_values: dict[str, list[float]]


def aggregated_plot_title(metric_name: str) -> str:
    generator = config.missing_generator
    parts = [f"geometry: {benchmark_geometry_type}"]
    if isinstance(generator, Mar):
        parts.append("missingness: MAR")
        parts.append(f"chunk: [{generator.chunk_start}, {generator.chunk_end}]")
    elif isinstance(generator, Mcar):
        parts.append("missingness: MCAR")
    else:
        parts.append(f"missingness: {type(generator).__name__}")
    missing_values = generator.missing_values
    if isinstance(missing_values, Rate):
        parts.append(f"rate: {missing_values.nb:.1%}")
    else:
        parts.append(f"missing count: {missing_values}")
    return (
        f"Aggregated Mean and Standard Deviation of {metric_name} "
        f"over {iterations} iterations "
        f"({', '.join(parts)})"
    )


def plot_aggregated_results_per_size(
    means_map: dict[str, list[float]],
    stds_map: dict[str, list[float]],
    metric_name: str,
    sample_sizes: list[int],
    block: bool = True,
) -> None:
    methods = list(means_map.keys())

    # Colors according to reference image - fixed order
    methods_order = [
        "KnnSampler",
        "RandomForestImputer",
        "KnnImputer",
        "KNNxKDEImputer",
        "LinearImputer",
    ]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ffb6c1", "#d4af37"]
    color_map = dict(zip(methods_order, colors, strict=False))

    x_indices = np.arange(len(sample_sizes))
    plt.figure(figsize=(12, 6))

    offset = 0.1
    for i, method in enumerate(methods):
        x_values = x_indices + (i - len(methods) / 2 + 0.5) * offset
        mean_values = means_map[method]
        std_values = stds_map[method]
        color = color_map.get(method, "black")

        plt.errorbar(
            x_values,
            mean_values,
            yerr=std_values,
            fmt="o",
            capsize=5,
            elinewidth=2,
            markeredgewidth=2,
            label=method,
            color=color,
        )

    plt.xlabel("Sample Size")
    plt.ylabel("Mean Value")
    plt.title(aggregated_plot_title(metric_name))
    plt.legend(
        loc="upper center",
        fontsize=12,
        bbox_to_anchor=(0.5, -0.1),
        ncol=round(len(methods) / 2.0),
    )
    plt.grid(True)
    plt.xticks(x_indices, [*map(str, sample_sizes)])
    plt.tight_layout()
    plt.show(block=block)


def create_data_preparator(sample_size):
    # Use a synthetic ring data generator as in the reference code
    # Enable Excel round-trip to mimic the original script I/O behavior
    return DataGenerator(
        linear_interpolation_ratio=1.0,
        sample_size=sample_size,
        missing_generator=config.missing_generator,
        geometry_type=benchmark_geometry_type,
    )


# ------------------------------------------------- Evaluation Metrics -------------------------------------------------


def evaluate_imputers(
    imputers: list[Imputer], data: DataFrameMLData, actual_values: pd.Series
) -> tuple[ImputersResults, ImputersResults, ImputersResults, ImputersResults]:
    """Get results for each imputer for each sample_size
    Args:
        imputers (list[Imputer]): _description_

    Returns:
        tuple[dict[Imputer, np.floating], dict[Imputer, np.floating], dict[Imputer, float]]: _description_
    """
    imputer_results: dict[Imputer, tuple[pd.DataFrame, float]] = do_imputation(
        *imputers, iterations=None
    )
    result_contexts: dict[Imputer, ImputationContext] = {
        imputer: ImputationContext.create_imputation_context(
            data, actual_values, df, et
        )
        for imputer, (df, et) in imputer_results.items()
    }

    rmse_values: dict[Imputer, float] = {}
    energy_distances: dict[Imputer, np.floating] = {}
    p_values: dict[Imputer, float] = {}
    execution_times: dict[Imputer, float] = {}
    for imputer, context in result_contexts.items():
        rmse_value = math.sqrt(
            mean_squared_error(
                context.actual_data[context.dataset_descriptor.target_column],
                context.predicted_data[context.dataset_descriptor.target_column],
            )
        )
        energy_distance = multivariate_energy_distance(
            context.predicted_data[context.dataset_descriptor.target_column]
            .to_numpy()
            .reshape(-1, 1),
            context.actual_data[context.dataset_descriptor.target_column]
            .to_numpy()
            .reshape(-1, 1),
        )
        Z = pd.concat(
            [
                context.predicted_data[context.dataset_descriptor.target_column],
                context.actual_data[context.dataset_descriptor.target_column],
            ],
            ignore_index=True,
        ).to_frame()
        p_value = calculate_p_value(
            permutation_test(Z, config.n_permutations), energy_distance
        )

        rmse_values[imputer] = rmse_value
        energy_distances[imputer] = energy_distance
        p_values[imputer] = p_value
        if (et := context.execution_time) is None:
            raise ValueError("execution time was not measured")
        execution_times[imputer] = et

    return rmse_values, energy_distances, p_values, execution_times


def benchmark_for_seed(
    seed: int | None = None,
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    list[dict[str, str]],
]:
    if seed is not None:
        np.random.seed(seed)

    rmse_values = defaultdict(list)
    ed_values = defaultdict(list)
    p_values = defaultdict(list)
    et_table_data: list[dict[str, str]] = []

    if REGENERATE_DATA_PER_METHOD:
        # For each sample size and each imputer, regenerate the dataset and evaluate only that method
        for _, sample_size in (
            bar := tqdm(
                enumerate(config.sample_sizes),
                total=len(config.sample_sizes),
                leave=False,
            )
        ):
            bar.set_description(
                f"executing for {sample_size = } (per-method regeneration)"
            )
            for imputer_cls, params in imputer_classes.items():
                # Fresh dataset for this method
                data_preparator = create_data_preparator(sample_size)
                prepared_data = data_preparator.prepare_data()
                actual_values = data_preparator.actual_values

                # Instantiate only the current imputer
                imputers = [*instantiate_imputers({imputer_cls: params}, prepared_data)]

                rmse_vals, eds, ps, ets = evaluate_imputers(
                    imputers, prepared_data, actual_values
                )

                # There is only one imputer in these dicts
                imputer_obj = next(iter(rmse_vals.keys()))
                name = imputer_obj.get_name()

                rmse_values[name].append(float(rmse_vals[imputer_obj]))
                ed_values[name].append(float(eds[imputer_obj]))
                p_values[name].append(float(ps[imputer_obj]))

                et = ets[imputer_obj]
                et_table_data.append(
                    {
                        "Imputer": name,
                        f"sample size of {sample_size}": f"{float(et):.2f}",
                    }
                )
    else:
        # Original benchmark behavior: one dataset per sample size, evaluate all imputers together
        results: dict[
            int,
            tuple[ImputersResults, ImputersResults, ImputersResults, ImputersResults],
        ] = {}
        for _, sample_size in (
            bar := tqdm(
                enumerate(config.sample_sizes),
                total=len(config.sample_sizes),
                leave=False,
            )
        ):
            bar.set_description(f"executing for {sample_size = }")
            data_preparator = create_data_preparator(sample_size)
            prepared_data = data_preparator.prepare_data()
            actual_values = data_preparator.actual_values

            imputers = [
                *instantiate_imputers(
                    imputer_classes,
                    prepared_data,
                )
            ]

            results[sample_size] = evaluate_imputers(
                imputers, prepared_data, actual_values
            )

        et_values = defaultdict(list)
        for _, (rmse_vals, eds, ps, ets) in results.items():
            for imp, result in rmse_vals.items():
                rmse_values[imp.get_name()].append(float(result))

            for imp, result in eds.items():
                ed_values[imp.get_name()].append(float(result))

            for imp, result in ps.items():
                p_values[imp.get_name()].append(float(result))

            for imp, result in ets.items():
                et_values[imp.get_name()].append(float(result))

        for imputer_name, execution_times in et_values.items():
            et_table_data.append(
                {"Imputer": str(imputer_name)}
                | dict(
                    zip(
                        map(lambda size: f"sample size of {size}", config.sample_sizes),
                        map(lambda et: f"{et:.2f}", execution_times),
                        strict=True,
                    )
                )
            )

    return (
        dict(rmse_values),
        dict(ed_values),
        dict(p_values),
        et_table_data,
    )


def benchmark():
    # Measure actual execution time
    start_time = time.perf_counter()

    # Collect all results from all iterations (per-method, per-sample-size)
    agg_rmse: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    agg_ed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    agg_p: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    results_per_iteration: dict[
        tuple[int, int],
        tuple[
            dict[str, list[float]],
            dict[str, list[float]],
            dict[str, list[float]],
            list[dict[str, str]],
        ],
    ] = {}
    base_seed = np.random.randint(1, 1 * 10**8)

    for iteration in (bar := trange(iterations)):
        seed = base_seed + iteration
        bar.set_description(f"iteration : {iteration}, seed : {seed}")
        rmse_values, ed_values, p_values, et_table_data = benchmark_for_seed(seed)
        results_per_iteration[iteration, seed] = (
            rmse_values,
            ed_values,
            p_values,
            et_table_data,
        )

        # Aggregate results per method and per sample size across iterations
        for method, values in rmse_values.items():
            for idx, sample_size in enumerate(config.sample_sizes):
                if idx < len(values):
                    agg_rmse[method][sample_size].append(values[idx])
        for method, values in ed_values.items():
            for idx, sample_size in enumerate(config.sample_sizes):
                if idx < len(values):
                    agg_ed[method][sample_size].append(values[idx])
        for method, values in p_values.items():
            for idx, sample_size in enumerate(config.sample_sizes):
                if idx < len(values):
                    agg_p[method][sample_size].append(values[idx])

        # Display results for this iteration
        print(f"\n{'=' * 40}\n")
        print(f"RESULTS:\nIteration: {iteration}, Seed: {seed}\n")

        print("RMSE Values:")
        pprint(rmse_values)

        print("\nEnergy Distance Values:")
        pprint(ed_values)

        print("\nP-values:")
        pprint(p_values)

    # Build per-method arrays of means/stds aligned with sample_sizes
    def build_mean_std_maps(
        agg: dict[str, dict[int, list[float]]],
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        means_map: dict[str, list[float]] = {}
        stds_map: dict[str, list[float]] = {}
        for method, per_size in agg.items():
            means: list[float] = []
            stds: list[float] = []
            for s in config.sample_sizes:
                arr = per_size.get(s, [])
                means.append(float(np.mean(arr)) if len(arr) > 0 else float("nan"))
                stds.append(float(np.std(arr)) if len(arr) > 0 else float("nan"))
            means_map[method] = means
            stds_map[method] = stds
        return means_map, stds_map

    rmse_means_map, rmse_stds_map = build_mean_std_maps(agg_rmse)
    ed_means_map, ed_stds_map = build_mean_std_maps(agg_ed)
    p_means_map, p_stds_map = build_mean_std_maps(agg_p)

    # Final aggregated plots with proper error bars across iterations
    plot_aggregated_results_per_size(
        rmse_means_map, rmse_stds_map, "RMSE", config.sample_sizes, block=False
    )
    plot_aggregated_results_per_size(
        ed_means_map, ed_stds_map, "Energy Distance", config.sample_sizes, block=False
    )
    plot_aggregated_results_per_size(
        p_means_map, p_stds_map, "P-Value", config.sample_sizes, block=True
    )

    # Calculate and display only the total execution time
    end_time = time.perf_counter()
    total_execution_time = end_time - start_time

    print(f"\n{'=' * 60}")
    print("TOTAL EXECUTION TIME")
    print("=" * 60)

    minutes = total_execution_time / 60
    hours = minutes / 60

    if hours >= 1:
        time_str = f"{total_execution_time:.0f}s (~{hours:.1f}h)"
    elif minutes >= 1:
        time_str = f"{total_execution_time:.0f}s (~{minutes:.1f}min)"
    else:
        time_str = f"{total_execution_time:.0f}s"

    print(f"Total execution time: {time_str}")
    print("=" * 60)


def main():
    profiler = Profiler([evaluate_imputers, benchmark_for_seed])
    profiler.start_profile(benchmark)
    stats_file = Path("./assets/profiling/benchmark_all_results.txt")
    profiler.print_stats(stats_file)


if __name__ == "__main__":
    main()
