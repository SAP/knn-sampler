import pandas as pd

import assets.config as conf
from assets.profiling.line_profile import Profiler
from src.imputation import (
    Imputer,
)
from src.utils import do_imputation, do_plot_results, instantiate_imputers


def main():
    prepared_data, actual_values = conf.prepare_data_from_config()
    imputers = instantiate_imputers(
        conf.imputers,
        prepared_data,
    )
    results: dict[Imputer, tuple[pd.DataFrame, float]]
    if conf.profile:
        profiler = Profiler([imp_class._execute for imp_class in conf.imputers])
        results = profiler.start_profile(do_imputation, *imputers, iterations=None)
        if conf.profile_dump_path is not None:
            profiler.dump_stats(conf.profile_dump_path)
        if conf.print_profile_stats:
            profiler.print_stats()

    else:
        results = do_imputation(*imputers)
    print("\nImputation done, plotting results...")
    if conf.plot_results:
        do_plot_results(
            results,  # type: ignore
            conf.missing_values,
            prepared_data,
            actual_values,
            sample_size=conf.data_preparator.sample_size,
        )


if __name__ == "__main__":
    main()
