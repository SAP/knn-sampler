import configparser
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import (
    DataGenerator,
    DataPreparator,
    ExcelDataPreparator,
    GeometryType,
    Mar,
    Mcar,
    MissingDataGenerator,
    Rate,
)
from src.data_preparation.dataset_description import ExcelDatasetDescriptor
from src.imputation import (
    Imputer,
    KnnImputer,
    KnnSampler,
    LinearImputer,
    RandomForestImputer,
)
from src.imputation.knnxkdeimputer import KNNxKDEImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.conf"
DEFAULT_HYPERPARAMETERS_PATH = Path(__file__).resolve().parent / "hyperparameters.conf"

_used_imputers: list[type[Imputer]] = [
    KnnImputer,
    KnnSampler,
    LinearImputer,
    RandomForestImputer,
    KNNxKDEImputer,
]

_str_to_imputer: dict[str, type[Imputer]] = {
    imp.__name__: imp for imp in _used_imputers
}


@dataclass
class Config:
    """Parsed configuration for the main imputation pipeline.

    Attributes
    ----------
    imputers : dict[type[Imputer], dict[str, Any]]
        Imputer classes to run and their constructor hyperparameters.
    plot_results : bool
        Whether to plot imputation results.
    profile : bool
        Whether to run imputations under the line profiler.
    print_profile_stats : bool
        Whether to print line profiling statistics.
    profile_dump_path : Path | None
        File to dump line profiling statistics to (created on demand), or None.
    data_preparator : DataPreparator
        Preparator producing the dataset to impute.
    missing_values : Rate | int
        Missing-value setting used by the preparator (rate or count).
    """

    imputers: dict[type[Imputer], dict[str, Any]]
    plot_results: bool
    profile: bool
    print_profile_stats: bool
    profile_dump_path: Path | None
    data_preparator: DataPreparator
    missing_values: Rate | int

    def prepare_data_from_config(self) -> tuple[DataFrameMLData, pd.Series]:
        """Prepare the configured dataset.

        Returns
        -------
        tuple[DataFrameMLData, pd.Series]
            Prepared ML data and the actual values of the removed entries.
        """
        print(
            f"preparing data with {type(self.data_preparator).__name__}: \n"
            f"{self.data_preparator}\n"
        )
        prepared_data = self.data_preparator.prepare_data()
        return prepared_data, self.data_preparator.actual_values


def _strip_quotes(value: str) -> str:
    """Strip one enclosing pair of single or double quotes if present."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _resolve_path(raw: str) -> Path:
    """Resolve a config path: strip enclosing quotes, anchor relative paths
    to the project root."""
    path = Path(_strip_quotes(raw))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_sample_size(fields: configparser.SectionProxy, name: str) -> int | None:
    """Parse the optional sample_size option (blank or absent means None)."""
    raw = fields.get("sample_size")
    if raw is None or raw.strip() == "":
        return None
    try:
        sample_size = int(raw)
    except ValueError as e:
        raise ValueError(
            f"sample_size should be an integer or left blank, got '{raw}'"
            f" for dataset {name}"
        ) from e
    if sample_size < 1:
        raise ValueError(f"sample_size should be an integer in [1, +∞[, got '{raw}'")
    return sample_size


def _parse_missing_values(fields: configparser.SectionProxy, name: str) -> Rate | int:
    """Parse missing_rate (wins when set) or missing_values (count)."""
    rate_raw = fields.get("missing_rate")
    if rate_raw:
        try:
            return Rate(float(rate_raw))
        except ValueError as e:
            raise ValueError(
                f"missing_rate should be a float in [0, 1], got '{rate_raw}'"
                f" for dataset {name}"
            ) from e
    values_raw = fields.get("missing_values")
    if values_raw is None:
        raise ValueError(
            f"dataset {name} must define missing_rate in [0, 1]"
            " or missing_values as a count"
        )
    try:
        return int(values_raw)
    except ValueError as e:
        raise ValueError(
            f"missing_values should be an integer, got '{values_raw}'"
            f" for dataset {name}"
        ) from e


def _parse_random_state(fields: configparser.SectionProxy, name: str) -> int | None:
    """Parse the optional random_state option (blank or absent means None)."""
    raw = fields.get("random_state")
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(
            f"random_state should be an integer or left blank, got '{raw}'"
            f" for dataset {name}"
        ) from e


def _extract_data_preparator(
    fields: configparser.SectionProxy, name: str
) -> tuple[DataPreparator, Rate | int]:
    """Build the data preparator described by a dataset config section."""
    sample_size = _parse_sample_size(fields, name)
    missing_values = _parse_missing_values(fields, name)
    random_state = _parse_random_state(fields, name)

    missing_generator: MissingDataGenerator
    match fields.get("missing_type"):
        case "MCAR":
            missing_generator = Mcar(missing_values, random_state=random_state)
        case "MAR":
            try:
                chunk_start = float(fields["MAR_chunk_start"])
                chunk_end = float(fields["MAR_chunk_end"])
            except KeyError as e:
                raise ValueError(
                    f"MAR missing type requires MAR_chunk_start and MAR_chunk_end"
                    f" for dataset {name}"
                ) from e
            missing_generator = Mar(
                chunk_start,
                chunk_end,
                missing_values=missing_values,
                random_state=random_state,
            )
        case _:
            raise ValueError(f"missing type is not defined for dataset {name}")

    match fields.get("type"):
        case "generated":
            if sample_size is None:
                raise ValueError(
                    f"sample_size is not defined or set to 0 for dataset {name}"
                )
            return DataGenerator(
                linear_interpolation_ratio=float(fields["linear_interpolation_ratio"]),
                sample_size=sample_size,
                missing_generator=missing_generator,
                # geometry_type is a free-form string here; DataGenerator's
                # constructor rejects values outside GEOMETRY_TYPES.
                geometry_type=cast(GeometryType, fields.get("geometry_type", "linear")),
                random_state=random_state,
            ), missing_values
        case "excel":
            return ExcelDataPreparator(
                dataset_descriptor=ExcelDatasetDescriptor(
                    fields["input"],
                    fields["target"],
                    str(_resolve_path(fields["path"])),
                    fields["sheet"],
                ),
                sample_size=sample_size,
                missing_generator=missing_generator,
                random_state=random_state,
            ), missing_values

    raise ValueError(
        f"type parameter should be in <generated | excel> for dataset {name},"
        f" got '{fields.get('type')}'"
    )


def load_config(
    config_path: Path | None = None,
    hyperparameters_path: Path | None = None,
) -> Config:
    """Load the pipeline configuration from files.

    Parameters
    ----------
    config_path : Path | None
        Main configuration file; defaults to assets/config.conf.
    hyperparameters_path : Path | None
        Imputer hyperparameters file; defaults to assets/hyperparameters.conf.

    Returns
    -------
    Config
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If either configuration file cannot be read.
    ValueError
        On unrecognized imputers, invalid option values or dataset settings.
    """
    conf = configparser.ConfigParser()
    if not conf.read(config_path or DEFAULT_CONFIG_PATH):
        raise FileNotFoundError(
            f"config file not found or unreadable: {config_path or DEFAULT_CONFIG_PATH}"
        )

    imputer_conf = configparser.ConfigParser()
    # disable automatic option name lowering: hyperparameter names are case sensitive
    imputer_conf.optionxform = lambda optionstr: optionstr  # type: ignore[assignment]
    if not imputer_conf.read(hyperparameters_path or DEFAULT_HYPERPARAMETERS_PATH):
        raise FileNotFoundError(
            "hyperparameters file not found or unreadable:"
            f" {hyperparameters_path or DEFAULT_HYPERPARAMETERS_PATH}"
        )

    imputers_str: list[str] = [
        i.strip()
        for i in conf["imputation_algorithms"]["imputation_algorithms"].split(",")
        if i.strip() != ""
    ]

    for imp_str in imputers_str:
        if imp_str not in (known := _str_to_imputer.keys()):
            raise ValueError(
                f"Unrecognized imputer :'{imp_str}',"
                f" Possible imputers are {', '.join(repr(i) for i in known)}"
            )

    imputers_types: list[type[Imputer]] = [_str_to_imputer[i] for i in imputers_str]

    plot_results = conf["plot"]["plot_results"] == "True"
    profile = conf["profile"]["profile"] == "True"
    raw_dump_path = conf["profile"].get("dump_path", "").strip()
    # The file does not have to exist: dump_stats creates it on demand.
    profile_dump_path = _resolve_path(raw_dump_path) if raw_dump_path else None
    print_profile_stats = conf["profile"].get("print_stats") == "True"
    used_dataset_config = conf["dataset_selection"]["used_dataset"]
    try:
        dataset_fields = conf[used_dataset_config]
    except KeyError as e:
        raise ValueError(
            f"dataset section '{used_dataset_config}' is not defined"
            f" (used_dataset in [dataset_selection])"
        ) from e
    data_preparator, missing_values = _extract_data_preparator(
        dataset_fields, used_dataset_config
    )

    imputer_args: list[dict[str, Any]] = []

    allowed_types: dict[str, Callable] = {
        "int": int,
        "int[]": lambda val: [int(v.strip()) for v in val.split(",")],
        "float": float,
        "bool": lambda val: val.lower() in ("true", "1", "yes", "on"),
        "str": str,
    }

    for istr in imputers_str:
        options: dict[str, Any] = dict(dict(imputer_conf).get(istr) or ())
        for o_name, val in options.items():
            if o_name == "ml_data":
                raise ValueError("ml_data is not configurable option")
            if len(splitted := val.split(":")) != 2:
                raise ValueError(
                    f"Format error for option '{o_name}', '{val}'"
                    f" should be of format '<value> : <type>'"
                )
            splitted = [i.strip() for i in splitted]
            value, value_type = splitted
            if (real_type := allowed_types.get(value_type)) is None:
                raise ValueError(
                    f"Unknown type for option '{o_name}', got '{value_type}'.\n"
                    f"Allowed types are"
                    f" {', '.join(repr(i) for i in allowed_types.keys())}"
                )
            try:
                real_value = real_type(value)
            except Exception as e:
                raise ValueError(
                    f"Could not convert '{value}' into a '{value_type}'"
                ) from e
            options[o_name] = real_value
        lk, uk = "lower_percentiles", "upper_percentiles"

        if lk in options.keys() and uk not in options.keys():
            options[uk] = [100 - i for i in options[lk]]

        if uk in options.keys() and lk not in options.keys():
            options[lk] = [100 - i for i in options[uk]]

        imputer_args.append(options)

    for section in dict(imputer_conf).keys():
        if section != "DEFAULT" and section not in imputers_str:
            warnings.warn(
                f"Warning : '{section}' is not a configurable imputer", stacklevel=2
            )

    return Config(
        imputers=dict(zip(imputers_types, imputer_args, strict=True)),
        plot_results=plot_results,
        profile=profile,
        print_profile_stats=print_profile_stats,
        profile_dump_path=profile_dump_path,
        data_preparator=data_preparator,
        missing_values=missing_values,
    )
