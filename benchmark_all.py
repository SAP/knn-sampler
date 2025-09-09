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

from assets.profiling.line_profile import Profiler
from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import ExcelDataPreparator, Mar, Mcar
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


mar_config = MissingConfig(Mar(0.5, 1.5, 200), [2000, 3000, 4000, 5000, 6000], 5000)
mcar_config = MissingConfig(Mcar(1000), [3000, 5000, 7000, 9000, 11000], 5000)

###### configuration selection ######
config: MissingConfig = mar_config
#####################################

imputer_classes: dict[type[Imputer], dict[str, typing.Any]] = {
    KnnSampler: {
        "upper_percentiles": [10, 20],
        "lower_percentiles": [90, 80],
        "compute_bounds": True,
    },
    KnnImputer: {"n_neighbors": 5},
    RandomForestImputer: {},
    KNNxKDEImputer: {},
    MICEImputer: {"n_neighbors": 5},
}

## KnnSampler config ##
iterations = 1
#######################

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
    mean_values: dict[str, float],
    std_values: dict[str, float],
    metric_name: str,
    sample_sizes: list[int],
    block: bool,
    seed: int | None = None,
) -> None:
    methods = list(mean_values.keys())
    colors: list[str] = [
        "green",
        "cyan",
        "purple",
        "blue",
        "pink",
        "orange",
        "yellow",
    ]
    offset = 0.4
    x_indices = np.arange(len(sample_sizes))
    plt.figure(figsize=(10, 6))

    for i, method in enumerate(methods):
        mean_list = [mean_values[method]] * len(sample_sizes)
        std_list = [std_values[method]] * len(sample_sizes)
        x_values = x_indices + (i - len(methods) / 2) * offset
        plt.errorbar(
            x_values,
            mean_list,
            yerr=std_list,
            fmt="o",
            capsize=5,
            elinewidth=2,
            markeredgewidth=2,
            label=f"{method} {f'for seed {seed}' if seed is not None else ''}",
            color=colors[i] if i < len(colors) else "black",
        )

    plt.xlabel("Sample Size")
    plt.ylabel("Mean Value")
    plt.title(f"Mean and Standard Deviation of {metric_name}")
    plt.legend(loc="upper center", fontsize=16, bbox_to_anchor=(0.5, 1.15), ncol=6)
    plt.grid(True)
    plt.xticks(x_indices, [*map(str, sample_sizes)])
    plt.show(block=block)


def create_data_preparator(sample_size):
    return ExcelDataPreparator(
        Datasets.PUBLICATION_DATASET,
        config.missing_generator,
        sample_size,
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
            mean_squared_error(context.actual_data, context.predicted_data)
        )
        energy_distance = multivariate_energy_distance_imputed(
            context.predicted_data.to_numpy(), context.actual_data.to_numpy()
        )
        Z = pd.concat([context.predicted_data, context.actual_data], ignore_index=True)
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


def benchmark_for_seed(seed: int | None = None) -> Benchmark:
    if seed:
        np.random.seed(seed)
    results: dict[
        int, tuple[ImputersResults, ImputersResults, ImputersResults, ImputersResults]
    ] = {}
    for _, sample_size in (
        bar := tqdm(
            enumerate(config.sample_sizes), total=len(config.sample_sizes), leave=False
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

        results[sample_size] = evaluate_imputers(imputers, prepared_data, actual_values)

    rmse_values = defaultdict(list)
    ed_values = defaultdict(list)
    p_values = defaultdict(list)
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

    rmse_mean_values = {
        imputer: float(sum(values) / len(values))
        for imputer, values in rmse_values.items()
    }
    rmse_std_values = {
        imputer: float(np.std(np.array(values)))
        for imputer, values in rmse_values.items()
    }
    ed_mean_values = {
        imputer: float(sum(values) / len(values))
        for imputer, values in ed_values.items()
    }
    ed_std_values = {
        imputer: float(np.std(np.array(values)))
        for imputer, values in ed_values.items()
    }
    p_mean_values = {
        imputer: float(sum(values) / len(values))
        for imputer, values in p_values.items()
    }
    p_std_values = {
        imputer: float(np.std(np.array(values))) for imputer, values in p_values.items()
    }

    et_table_data: list[dict[str, str]] = []
    for imputer_, execution_times in et_values.items():
        et_table_data.append(
            {"Imputer": str(imputer_)}
            | dict(
                zip(
                    map(lambda size: f"sample size of {size}", config.sample_sizes),
                    map(lambda et: f"{et:.2f}", execution_times),
                    strict=True,
                )
            )
        )

    return (
        (rmse_mean_values, rmse_std_values),
        (ed_mean_values, ed_std_values),
        (p_mean_values, p_std_values),
        et_table_data,
    )


def benchmark():
    results_per_iteration: dict[tuple[int, int], Benchmark] = {}
    base_seed = np.random.randint(1, 1 * 10**8)
    for iteration in (bar := trange(iterations)):
        seed = base_seed + iteration
        bar.set_description(f"iteration : {iteration}, seed : {seed}")
        results_per_iteration[iteration, seed] = benchmark_for_seed(seed)
    for index, (
        (iteration, seed),
        (
            (rmse_mean_values, rmse_std_values),
            (ed_mean_values, ed_std_values),
            (p_mean_values, p_std_values),
            et_table_data,
        ),
    ) in enumerate(results_per_iteration.items()):
        print(f"\n{'=' * 40}\n")
        print(f"RESULTS:\nIteration: {iteration}, Seed: {seed}\n")

        print("RMSE Mean Values:")
        pprint(rmse_mean_values)
        print("\nRMSE Standard Deviation Values:")
        pprint(rmse_std_values)

        print("\nEnergy Distance Mean Values:")
        pprint(ed_mean_values)
        print("\nEnergy Distance Standard Deviation Values:")
        pprint(ed_std_values)

        print("\nP Distance Mean Values:")
        pprint(p_mean_values)
        print("\nP Distance Standard Deviation Values:")
        pprint(p_std_values)

        print(f"\n\nExecution times for seed: {seed}\n")
        print(
            tabulate(et_table_data, headers="keys", tablefmt="grid", numalign="center")
        )

        plot_results(
            rmse_mean_values,
            rmse_std_values,
            "RMSE",
            config.sample_sizes,
            block=False,
            seed=seed,
        )
        plot_results(
            ed_mean_values,
            ed_std_values,
            "Energy Distance",
            config.sample_sizes,
            block=False,
            seed=seed,
        )
        plot_results(
            p_mean_values,
            p_std_values,
            "P-value",
            config.sample_sizes,
            block=index == iterations - 1,
            seed=seed,
        )


def main():
    profiler = Profiler([evaluate_imputers, benchmark_for_seed])
    profiler.start_profile(benchmark)
    stats_file = Path("./assets/profiling/benchmark_all_results.txt")
    profiler.print_stats(stats_file)


if __name__ == "__main__":
    main()
