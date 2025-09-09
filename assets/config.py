import configparser
import typing
import warnings
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from src.data_preparation.data_description import DataFrameMLData
from src.data_preparation.data_preparation import (
    DataGenerator,
    DataPreparator,
    ExcelDataPreparator,
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

imputers: dict[type[Imputer], dict[str, typing.Any]]
plot_results: bool
profile: bool
print_profile_stats: bool
profile_dump_path: Path | None
data_preparator: DataPreparator
mar_chunk: dict[Literal["start"] | Literal["end"], float | None] = {}
missing_values: Rate | int
_used_imputers: list[type[Imputer]] = [
    KnnImputer,
    KnnSampler,
    LinearImputer,
    RandomForestImputer,
    KNNxKDEImputer,
]

_strToImputer = dict([(imp.__name__, imp) for imp in _used_imputers])


def get_field(fields, field_name):
    try:
        return fields[field_name]
    except KeyError:
        return None


def _extract_data_preparator(fields, name) -> tuple[DataPreparator, Rate | int]:
    if (sample_size_str := fields.get("sample_size")) is None:
        raise ValueError("Sample size is not defined")
    sample_size = int(sample_size_str)
    if sample_size < 1:
        raise ValueError("Sample size should be an integer in [1, +∞[")

    missing_values = (
        Rate(float(rate))
        if (rate := get_field(fields, "missing_rate"))
        else int(fields["missing_values"])
    )
    missing_generator: MissingDataGenerator
    match fields["missing_type"]:
        case "MCAR":
            missing_generator = Mcar(missing_values)
        case "MAR":
            missing_generator = Mar(
                float(fields["MAR_chunk_start"]),
                float(fields["MAR_chunk_end"]),
                missing_values=missing_values,
            )
        case _:
            raise ValueError("missing type is not defined")

    match fields["type"]:
        case "generated":
            if sample_size is None:
                raise ValueError("Sample size is not defined or set to 0")
            return DataGenerator(
                linear_interpolation_ratio=float(fields["linear_interpolation_ratio"]),
                sample_size=sample_size,
                missing_generator=missing_generator,
                geometry_type=fields.get("geometry_type", "linear"),
            ), missing_values
        case "excel":
            return ExcelDataPreparator(
                dataset_descriptor=ExcelDatasetDescriptor(
                    fields["input"], fields["target"], fields["path"], fields["sheet"]
                ),
                sample_size=sample_size,
                missing_generator=missing_generator,
            ), missing_values

    raise ValueError(
        f"type parameter should be in <generated | excel> for dataset {name}, got {fields['type']}"
    )


_firstImport = True
if _firstImport:
    _firstImport = False
    conf = configparser.ConfigParser()
    conf.read("assets/config.conf")

    imputer_conf = configparser.ConfigParser()
    # override auto option lowering :
    imputer_conf.optionxform = lambda optionstr: optionstr  # type: ignore
    imputer_conf.read("assets/hyperparameters.conf")

    imputers_str: list[str] = [
        i.strip()
        for i in conf["imputation_algorithms"]["imputation_algorithms"].split(",")
    ]

    for imp_str in imputers_str:
        if imp_str not in (imps := _strToImputer.keys()):
            raise ValueError(
                f"Unrecognized imputer :'{imp_str}', Possible imputers are {', '.join([repr(i) for i in imps])}"
            )

    imputers_types: list[typing.Any] = [_strToImputer[i.strip()] for i in imputers_str]  # type: ignore

    plot_results = conf["plot"]["plot_results"] == "True"
    profile = conf["profile"]["profile"] == "True"

    profile_dump_path = Path(conf["profile"].get("dump_path") or "")
    profile_dump_path = profile_dump_path if profile_dump_path.is_file() else None
    print_profile_stats = conf["profile"].get("print_stats") == "True"
    used_dataset_config = conf["dataset_selection"]["used_dataset"]
    data_preparator, missing_values = _extract_data_preparator(
        conf[used_dataset_config], used_dataset_config
    )

    imputer_args: list[dict[str, typing.Any]] = []

    allowed_types: dict[str, Callable] = {
        "int": int,
        "int[]": lambda val: [int(v.strip()) for v in val.split(",")],
        "float": float,
        "bool": bool,
    }

    for istr in imputers_str:
        options: dict[str, Any] = dict(dict(imputer_conf).get(istr) or ())
        for o_name, val in options.items():
            if o_name == "ml_data":
                raise ValueError("ml_data is not configurable option")
            if len(splitted := val.split(":")) != 2:
                raise ValueError(
                    f"Format error for option '{o_name}', '{val}' should be of format '<value> : <type>'"
                )
            splitted = [i.strip() for i in splitted]
            value, value_type = splitted
            if (real_type := allowed_types.get(value_type)) is None:
                raise ValueError(
                    f"Unknown type for option '{o_name}', got '{value_type}'.\nAllowed types are {', '.join([repr(i) for i in allowed_types.keys()])}"
                )
            try:
                real_value = real_type(value)
            except Exception as e:
                raise ValueError(
                    f"Could not convert '{value}' into a '{value_type}'"
                ) from e
            options[o_name] = real_value
        imputer_args.append(options)
        lk, uk = "lower_percentiles", "upper_percentiles"

        if lk in options.keys() and uk not in options.keys():
            options[uk] = [100 - i for i in options[lk]]

        if uk in options.keys() and lk not in options.keys():
            options[lk] = [100 - i for i in options[uk]]

    for section in dict(imputer_conf).keys():
        if section != "DEFAULT" and section not in imputers_str:
            warnings.warn(
                f"Warning : '{section}' is not a configurable imputer", stacklevel=1
            )
    imputers = dict(zip(imputers_types, imputer_args, strict=True))


def prepare_data_from_config() -> tuple[DataFrameMLData, pd.Series]:
    print(
        f"preparing data with {type(data_preparator).__name__}: \n{data_preparator}\n"
    )
    prepared_data = data_preparator.prepare_data()
    return prepared_data, data_preparator.actual_values
