import inspect
import time
from collections import defaultdict
from collections.abc import Generator, Mapping
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import Rate
from src.imputation import Imputer
from src.imputation.imputer import UncertaintyImputer, validate_random_state
from src.imputation_context import ImputationContext
from src.imputation_plot import ImputationPlot

# split_random_state() reserves the first two children of SeedSequence(seed)
# for preparation and missingness. Execution seeds use the following children:
# their assembled entropy ends with an index >= 2, so it cannot coincide with
# those streams for any seed magnitude.
_EXECUTION_SEED_OFFSET: int = 2


def _derive_execution_seeds(random_state: int | None, count: int) -> list[int | None]:
    """Derive one execution seed per iteration from an optional run seed.

    Parameters
    ----------
    random_state : int | None
        Run-level seed. None keeps every execution nondeterministic.
    count : int
        Number of iterations to derive seeds for.

    Returns
    -------
    list[int | None]
        One Python integer seed per iteration, or None values when unseeded.

    Raises
    ------
    ValueError
        If random_state is not a non-negative integer or None.
    """
    random_state = validate_random_state(random_state)
    if random_state is None:
        return [None] * count
    root = np.random.SeedSequence(random_state)
    children = root.spawn(_EXECUTION_SEED_OFFSET + count)
    return [
        int(child.generate_state(1)[0]) for child in children[_EXECUTION_SEED_OFFSET:]
    ]


@overload
def do_imputation(
    *imputers: Imputer,
    iterations: Literal[None] = None,
    random_state: int | None = None,
) -> dict[Imputer, tuple[pd.DataFrame, float]]: ...


@overload
def do_imputation(
    *imputers: Imputer,
    iterations: int,
    random_state: int | None = None,
) -> dict[Imputer, list[tuple[pd.DataFrame, float]]]: ...


def do_imputation(
    *imputers: Imputer,
    iterations: int | None = None,
    random_state: int | None = None,
) -> (
    dict[Imputer, tuple[pd.DataFrame, float]]
    | dict[Imputer, list[tuple[pd.DataFrame, float]]]
):
    """Run all given imputers, optionally with multiple imputations.

    Parameters
    ----------
    *imputers : Imputer
        Imputers to fit and execute.
    iterations : int | None
        Number of imputation iterations per imputer. None runs one execution.
    random_state : int | None
        Run-level seed for stochastic imputations. It is expanded into one
        independent seed per iteration, and every imputer receives the same
        per-iteration seed, so seed assignment does not depend on imputer
        order. Results are order-independent only for imputers that honor the
        seed without consuming global randomness. None leaves stochastic
        imputations unseeded.

    Returns
    -------
    dict[Imputer, tuple[pd.DataFrame, float]]
        If iterations is None, one timed result per imputer.
    dict[Imputer, list[tuple[pd.DataFrame, float]]]
        Otherwise, one timed result per imputer and iteration.
    """
    seed_count = 1 if iterations is None else max(iterations, 0)
    execution_seeds = _derive_execution_seeds(random_state, seed_count)
    if iterations is None:
        result: dict[Imputer, tuple[pd.DataFrame, float]] = {}
        for imputer in (bar := tqdm(imputers, leave=False)):
            bar.set_description(f"Fitting data for {imputer.get_name()}:")
            start = time.perf_counter()
            imputer.fit()
            bar.set_description(f"Imputing missing data with {imputer.get_name()}:")
            result[imputer] = (
                imputer.execute(execution_seeds[0]),
                time.perf_counter() - start,
            )
        return result
    results: dict[Imputer, list[tuple[pd.DataFrame, float]]] = defaultdict(list)
    for imputer in (bar := tqdm(imputers, leave=False)):
        bar.set_description(f"Fitting data for {imputer.get_name()}:")
        start_fit = time.perf_counter()
        imputer.fit()
        fit_time = time.perf_counter() - start_fit
        bar.set_description(f"Imputing missing data with {imputer.get_name()}:")
        for iteration in trange(iterations):
            start = time.perf_counter()
            res = imputer.execute(execution_seeds[iteration])
            results[imputer].append((res, fit_time + (time.perf_counter() - start)))
    return results


def do_plot_results(
    results: Mapping[Imputer, tuple[pd.DataFrame, float | None]],
    missing_values: int | Rate,
    data: DataFrameMLData,
    actual_values: pd.Series,
    block: bool = True,
    sample_size: int | None = None,
) -> None:
    """Plot imputation results for each imputer.

    The original-data figure is drawn once per call (for the first
    imputer), then each imputer gets its original-vs-imputed comparison
    plus uncertainty bounds when available; show() blocks only on the
    last imputer's figures.
    """
    items = results.items()
    for index, (imputer, (result, et)) in enumerate(items):
        context = ImputationContext.create_imputation_context(
            data, actual_values, result, et
        )
        imputation_plot = ImputationPlot(
            imputer.get_name(),
            context,
            sample_size,
            missing_values,
        )
        if index == 0:
            imputation_plot.plot_original()
            # imputation_plot.plot_original_scatterplot()
        imputation_plot.plot_original_imputed()
        # imputation_plot.plot_imputation_results()
        # imputation_plot.plot_imputed_accuracy()
        # imputation_plot.plot_local_distance()
        if isinstance(imputer, UncertaintyImputer):
            for (lp, up), (lb, ub) in imputer.bounds.items():
                imputation_plot.plot_uncertainty_results(
                    lower_percentile=lp,
                    upper_percentile=up,
                    lower_bounds=lb,
                    upper_bounds=ub,
                )
        imputation_plot.show(
            block if index == len(items) - 1 else False
        )  # block for the last imputer


def instantiate_imputers(
    imputer_classes: dict[type[Imputer], dict[str, Any]],
    data: DataFrameMLData,
) -> Generator[Imputer, Any, None]:
    for ic, kwargs in imputer_classes.items():
        try:
            yield ic(data, **kwargs)
        except TypeError as e:
            unrecognized_args = [
                arg
                for arg in kwargs
                if arg not in inspect.getfullargspec(ic.__init__)[0]
            ]
            raise ValueError(
                f"Wrong configuration arguments for {ic.__name__}, unexpected hyperparameter(s) : {', '.join(unrecognized_args)}"
            ) from e
