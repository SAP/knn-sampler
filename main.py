import pandas as pd

import assets.config as conf
from assets.profiling.line_profile import Profiler
from src.imputation import (
    Imputer,
)
from src.utils import do_imputation, do_plot_results, instantiate_imputers


def main() -> None:
    config = conf
    prepared_data, actual_values = config.prepare_data_from_config()
    imputers = instantiate_imputers(
        config.imputers,
        prepared_data,
    )
    results: dict[Imputer, tuple[pd.DataFrame, float]]
    if config.profile:
        profiler = Profiler([imp_class._execute for imp_class in config.imputers])
        results = profiler.start_profile(do_imputation, *imputers, iterations=None)
        if config.profile_dump_path is not None:
            profiler.dump_stats(config.profile_dump_path)
        if config.print_profile_stats:
            profiler.print_stats()

    else:
        results = do_imputation(*imputers)
    print("\nImputation done, plotting results...")
    if config.plot_results:
        do_plot_results(
            results,
            config.missing_values,
            prepared_data,
            actual_values,
            sample_size=len(prepared_data.df),
        )


if __name__ == "__main__":
    main()
