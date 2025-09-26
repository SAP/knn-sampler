import math
import typing
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from tabulate import tabulate  # type: ignore
from tqdm import tqdm, trange
import time
from assets.profiling.line_profile import Profiler
from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import ExcelDataPreparator, DataGenerator, Mar, Mcar, Rate
from src.data_preparation.dataset_description import (
    Datasets,
)
from src.imputation import KnnImputer, KnnSampler, RandomForestImputer
from src.imputation.imputer import Imputer
from src.imputation.knnxkdeimputer import KNNxKDEImputer
from src.imputation.miceimputer import MICEImputer
from src.imputation_context import ImputationContext
from src.stats_utils import (
    calculate_p_value,
    multivariate_energy_distance_imputed,
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
mcar_config = MissingConfig(Mcar(Rate(0.3)), [3000], 200)

###### configuration selection ######
config: MissingConfig = mcar_config
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
    MICEImputer: {"max_iter": 10},
}

## KnnSampler config ##
iterations = 5  # Aligned with Imputation-full-code (1).py range(2)
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


root_rmse_values: dict[str, list[np.floating | float]]
root_ed_values: dict[str, list[np.floating | float]]
root_p_values: dict[str, list[float]]
root_et_values: dict[str, list[float]]


def plot_results(
    values: dict[str, list[float]],
    metric_name: str,
    sample_sizes: list[int],
    block: bool,
    seed: int | None = None,
) -> None:
    methods = list(values.keys())

    # Colors according to reference image - Order: kNNSampler → Random Forest → kNNImputer → KNNxKDE → MICE
    methods_order = ["KnnSampler", "RandomForestImputer", "KnnImputer", "KNNxKDEImputer", "MICEImputer"]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ffb6c1", "#d4af37"]  # Blue, Green, Purple, Light pink, Beige/gold
    color_map = dict(zip(methods_order, colors))

    x_indices = np.arange(len(sample_sizes))
    plt.figure(figsize=(12, 6))

    offset = 0.1
    for i, method in enumerate(methods):
        x_values = x_indices + (i - len(methods) / 2 + 0.5) * offset
        mean_values = values[method]
        std_values = [np.std(values[method])] * len(sample_sizes)  # Calculate standard deviation

        # Use the corresponding color or a default color
        color = color_map.get(method, "black")

        plt.errorbar(
            x_values,
            mean_values,
            yerr=std_values,
            fmt="o",
            capsize=5,
            elinewidth=2,
            markeredgewidth=2,
            label=f"{method} {f'for seed {seed}' if seed is not None else ''}",
            color=color,
        )

    plt.xlabel("Sample Size")
    plt.ylabel(f"{metric_name} Value")
    plt.title(f"{metric_name} for Different Sample Sizes")
    # Legend on the right as in the reference image
    plt.legend(loc="center left", fontsize=10, bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.xticks(x_indices, [*map(str, sample_sizes)])
    plt.tight_layout()
    plt.show(block=block)


def plot_results_with_error_bars(
    means: dict[str, float],
    stds: dict[str, float],
    metric_name: str,
    sample_sizes: list[int],
) -> None:
    methods = list(means.keys())

    # Colors according to reference image - Order: kNNSampler → Random Forest → kNNImputer → KNNxKDE → MICE
    methods_order = ["KnnSampler", "RandomForestImputer", "KnnImputer", "KNNxKDEImputer", "MICEImputer"]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ffb6c1", "#d4af37"]  # Blue, Green, Purple, Light pink, Beige/gold
    color_map = dict(zip(methods_order, colors))

    x_indices = np.arange(len(sample_sizes))
    plt.figure(figsize=(12, 6))

    offset = 0.1
    for i, method in enumerate(methods):
        x_values = x_indices + (i - len(methods) / 2 + 0.5) * offset
        mean_value = [means[method]] * len(sample_sizes)
        std_value = [stds[method]] * len(sample_sizes)

        # Use the corresponding color or a default color
        color = color_map.get(method, "black")

        plt.errorbar(
            x_values,
            mean_value,
            yerr=std_value,
            fmt="o",
            capsize=5,
            elinewidth=2,
            markeredgewidth=2,
            label=method,
            color=color,
        )

    plt.xlabel("Sample Size")
    plt.ylabel(f"{metric_name} Value")
    plt.title(f"{metric_name} - Aggregated ({iterations} iterations)")
    # Legend on the right as in the reference image
    plt.legend(loc="center left", fontsize=10, bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.xticks(x_indices, [*map(str, sample_sizes)])
    plt.tight_layout()
    plt.show(block=True)


def plot_aggregated_results_per_size(
    means_map: dict[str, list[float]],
    stds_map: dict[str, list[float]],
    metric_name: str,
    sample_sizes: list[int],
    block: bool = True,
) -> None:
    methods = list(means_map.keys())

    # Colors according to reference image - fixed order
    methods_order = ["KnnSampler", "RandomForestImputer", "KnnImputer", "KNNxKDEImputer", "MICEImputer"]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ffb6c1", "#d4af37"]
    color_map = dict(zip(methods_order, colors))

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
    plt.ylabel(f"{metric_name} Value")
    plt.title(f"{metric_name} - Aggregated over iterations")
    plt.legend(loc="center left", fontsize=10, bbox_to_anchor=(1, 0.5))
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
        geometry_type="ring",
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
                context.predicted_data[context.dataset_descriptor.target_column]
            )
        )
        energy_distance = multivariate_energy_distance_imputed(
            context.predicted_data[context.dataset_descriptor.target_column].to_numpy().reshape(-1, 1),
            context.actual_data[context.dataset_descriptor.target_column].to_numpy().reshape(-1, 1)
        )
        Z = pd.concat([
            context.predicted_data[context.dataset_descriptor.target_column],
            context.actual_data[context.dataset_descriptor.target_column]
        ], ignore_index=True).to_frame()
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


def benchmark_for_seed(seed: int | None = None) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], list[dict[str, str]]]:
    if seed:
        np.random.seed(seed)

    rmse_values = defaultdict(list)
    ed_values = defaultdict(list)
    p_values = defaultdict(list)
    et_table_data: list[dict[str, str]] = []


    # TODO: make this cleaner
    if REGENERATE_DATA_PER_METHOD:
        # For each sample size and each imputer, regenerate the dataset and evaluate only that method
        for _, sample_size in (
            bar := tqdm(enumerate(config.sample_sizes), total=len(config.sample_sizes), leave=False)
        ):
            bar.set_description(f"executing for {sample_size = } (per-method regeneration)")
            for imputer_cls, params in imputer_classes.items():
                # Fresh dataset for this method
                data_preparator = create_data_preparator(sample_size)
                prepared_data = data_preparator.prepare_data()
                actual_values = data_preparator.actual_values

                # Instantiate only the current imputer
                imputers = [*instantiate_imputers({imputer_cls: params}, prepared_data)]

                rmse_vals, eds, ps, ets = evaluate_imputers(imputers, prepared_data, actual_values)

                # There is only one imputer in these dicts
                imputer_obj = next(iter(rmse_vals.keys()))
                name = imputer_obj.get_name()

                rmse_values[name].append(float(rmse_vals[imputer_obj]))
                ed_values[name].append(float(eds[imputer_obj]))
                p_values[name].append(float(ps[imputer_obj]))

                et = ets[imputer_obj]
                et_table_data.append(
                    {"Imputer": name, f"sample size of {sample_size}": f"{float(et):.2f}"}
                )
    else:
        # Original benchmark behavior: one dataset per sample size, evaluate all imputers together
        results: dict[
            int, tuple[ImputersResults, ImputersResults, ImputersResults, ImputersResults]
        ] = {}
        for _, sample_size in (
            bar := tqdm(enumerate(config.sample_sizes), total=len(config.sample_sizes), leave=False)
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

            results[sample_size] = evaluate_imputers(imputers, prepared_data, actual_values)

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


def estimate_total_time():
    """Estimate total execution time based on parameters"""

    # Base time per imputer (in seconds) for 1000 samples with 30% missing
    base_times = {
        "KnnSampler": 2.0,        # O(n²) for k-NN
        "KnnImputer": 0.5,        # Simpler
        "RandomForestImputer": 1.5, # O(n*log(n))
        "KNNxKDEImputer": 3.0,    # More complex with KDE
        "MICEImputer": 4.0,       # Iterative, slower
    }

    total_estimated_time = 0

    for sample_size in config.sample_sizes:
        missing_count = int(sample_size * 0.3)  # 30% missing

        # Scale factor based on algorithmic complexity
        scale_factor = (sample_size / 1000)

        sample_time = 0
        for imputer_name, base_time in base_times.items():
            # Different algorithmic complexities
            if "Knn" in imputer_name:
                # O(n²) for k-NN
                imputer_time = base_time * (scale_factor ** 1.5)
            elif "RandomForest" in imputer_name:
                # O(n*log(n)) for Random Forest
                imputer_time = base_time * scale_factor * math.log(scale_factor + 1)
            else:
                # O(n) linear for others
                imputer_time = base_time * scale_factor

            sample_time += imputer_time

        iteration_time = sample_time * iterations
        total_estimated_time += iteration_time

    # Add overhead time (permutation tests, plots, etc.)
    overhead_time = total_estimated_time * 0.2  # 20% overhead
    total_estimated_time += overhead_time

    return total_estimated_time


def benchmark():
    # Estimate total time (no display)
    estimated_time = estimate_total_time()

    # Measure actual execution time
    start_time = time.perf_counter()

    # Collect all results from all iterations (per-method, per-sample-size)
    agg_rmse: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    agg_ed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    agg_p: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    results_per_iteration: dict[tuple[int, int], tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], list[dict[str, str]]]] = {}
    base_seed = np.random.randint(1, 1 * 10**8)

    for iteration in (bar := trange(iterations)):
        seed = base_seed + iteration
        bar.set_description(f"iteration : {iteration}, seed : {seed}")
        rmse_values, ed_values, p_values, et_table_data = benchmark_for_seed(seed)
        results_per_iteration[iteration, seed] = (rmse_values, ed_values, p_values, et_table_data)

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


        # Create individual plots for this iteration as before
        # Display RMSE and Energy Distance non-blocking at each iteration
        plot_results(rmse_values, "RMSE", config.sample_sizes, block=False, seed=seed)
        plot_results(ed_values, "Energy Distance", config.sample_sizes, block=False, seed=seed)
        # Do NOT display the P-Value plot for the last iteration (it will be displayed blocking after all prints)
        if iteration < iterations - 1:
            plot_results(p_values, "P-Value", config.sample_sizes, block=False, seed=seed)

    # Build per-method arrays of means/stds aligned with sample_sizes
    def build_mean_std_maps(agg: dict[str, dict[int, list[float]]]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
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
    plot_aggregated_results_per_size(rmse_means_map, rmse_stds_map, "RMSE", config.sample_sizes, block=False)
    plot_aggregated_results_per_size(ed_means_map, ed_stds_map, "Energy Distance", config.sample_sizes, block=False)
    plot_aggregated_results_per_size(p_means_map, p_stds_map, "P-Value", config.sample_sizes, block=False)


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

    # Display the last blocking plot after all prints
    if iterations > 0:
        # Get results from the last iteration
        last_iteration_key = (iterations - 1, base_seed + iterations - 1)
        if last_iteration_key in results_per_iteration:
            last_rmse, last_ed, last_p, _ = results_per_iteration[last_iteration_key]
            # Only the last P-Value plot is blocking
            plot_results(last_p, "P-Value", config.sample_sizes, block=True, seed=base_seed + iterations - 1)


def main():
    profiler = Profiler([evaluate_imputers, benchmark_for_seed])
    profiler.start_profile(benchmark)
    stats_file = Path("./assets/profiling/benchmark_all_results.txt")
    profiler.print_stats(stats_file)


if __name__ == "__main__":
    main()
