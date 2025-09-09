import inspect
import time
from collections import defaultdict
from typing import Any, Generator, Literal, Union, overload

import pandas as pd
from tqdm import tqdm, trange

from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import Rate
from src.imputation import Imputer
from src.imputation.imputer import UncertaintyImputer
from src.imputation_context import ImputationContext
from src.imputation_plot import ImputationPlot


@overload
def do_imputation(
    *imputers: Imputer, iterations: Literal[None] = None
) -> dict[Imputer, tuple[pd.DataFrame, float]]: ...


@overload
def do_imputation(
    *imputers: Imputer, iterations: int
) -> dict[Imputer, list[tuple[pd.DataFrame, float]]]: ...


def do_imputation(
    *imputers: Imputer, iterations: Union[int, None] = None
) -> Union[
    dict[Imputer, tuple[pd.DataFrame, float]],
    dict[Imputer, list[tuple[pd.DataFrame, float]]],
]:
    """
    Gives imputation results for all given imputers, for each iterations <br>
    returns -> dict[Imputer, list[tuple[pd.DataFrame, float]]] if iterations is set <br>
    returns -> dict[Imputer, tuple[pd.DataFrame, float]] else
    """
    if iterations is None:
        result: dict[Imputer, tuple[pd.DataFrame, float]] = {}
        for imputer in (bar := tqdm(imputers, leave=False)):
            bar.set_description(f"Fitting data for {imputer.get_name()}:")
            start = time.perf_counter()
            imputer.fit()
            bar.set_description(f"Imputing missing data with {imputer.get_name()}:")
            result[imputer] = imputer.execute(), time.perf_counter() - start
        return result
    results: dict[Imputer, list[tuple[pd.DataFrame, float]]] = defaultdict(list)
    for imputer in (bar := tqdm(imputers, leave=False)):
        bar.set_description(f"Fitting data for {imputer.get_name()}:")
        start_fit = time.perf_counter()
        imputer.fit()
        fit_time = time.perf_counter() - start_fit
        bar.set_description(f"Imputing missing data with {imputer.get_name()}:")
        for _ in trange(iterations):
            start = time.perf_counter()
            res = imputer.execute()
            results[imputer].append((res, fit_time + (time.perf_counter() - start)))
    return results


plot_original = False


def do_plot_results(
    results: dict[Imputer, tuple[pd.DataFrame, Union[float, None]]],
    missing_values: Union[int, Rate],
    data: DataFrameMLData,
    actual_values: pd.Series,
    block: bool = True,
    sample_size: Union[int, None] = None,
):
    global plot_original
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
        if not plot_original:
            imputation_plot.plot_original()
            # imputation_plot.plot_original_scatterplot()
            plot_original = True
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
