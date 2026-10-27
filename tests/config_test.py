import importlib.util
import re
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import assets.config as configuration
from src.data_preparation.data_preparation import (
    DataGenerator,
    DataPreparator,
    Mar,
    Mcar,
    NoAdditionalMissingValues,
    Rate,
    split_random_state,
)
from src.imputation import (
    KnnImputer,
    KnnSampler,
    LinearImputer,
    RandomForestImputer,
)
from src.imputation.knnxkdeimputer import KNNxKDEImputer

KNOWN_IMPUTERS = {
    KnnImputer,
    KnnSampler,
    LinearImputer,
    RandomForestImputer,
    KNNxKDEImputer,
}


@pytest.mark.parametrize("sample_size", ["", "4"])
def test_config_selects_requested_excel_sheet(
    tmp_path, multi_sheet_workbook, sample_size
):
    conf = (
        base_conf(
            extra=(
                f'\npath = "{multi_sheet_workbook}"\nsheet = selected\ninput = input\ntarget = target\nrandom_state = 42'
            )
        )
        .replace("type = generated", "type = excel")
        .replace("sample_size = 100", f"sample_size = {sample_size}")
    )
    preparator = load_from_conf(tmp_path, conf).data_preparator
    prepared = preparator.prepare_data().df
    assert len(prepared) == (int(sample_size) if sample_size else 8)
    assert prepared.Y.dropna().between(100, 107).all()
    assert preparator.actual_values.between(100, 107).all()


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR"])
@pytest.mark.parametrize("dataset_type", ["generated", "excel"])
def test_config_splits_seed_and_replays_data(tmp_path, mechanism, dataset_type):
    conf = base_conf(extra="random_state = 42\nMAR_chunk_start = -1\nMAR_chunk_end = 1")
    conf = conf.replace("missing_type = MCAR", f"missing_type = {mechanism}")
    if dataset_type == "excel":
        source = tmp_path / "source.xlsx"
        pd.DataFrame({"X": np.linspace(-2, 2, 120), "Y": range(120)}).to_excel(
            source, index=False
        )
        conf = conf.replace("type = generated", "type = excel")
        conf += f"\npath = {source}\nsheet = Sheet1\ninput = X\ntarget = Y\n"
    config = load_from_conf(tmp_path, conf)
    preparator = config.data_preparator
    assert isinstance(preparator.missing_generator, (Mcar, Mar))
    assert preparator.random_state != preparator.missing_generator.random_state
    first = preparator.prepare_data()
    mask = preparator.missing_indexes.copy()
    second = preparator.prepare_data()
    pd.testing.assert_frame_equal(first.df, second.df)
    np.testing.assert_array_equal(mask, preparator.missing_indexes)
    replay = load_from_conf(tmp_path, conf).data_preparator.prepare_data()
    pd.testing.assert_frame_equal(first.df, replay.df)
    changed = load_from_conf(
        tmp_path, conf.replace("random_state = 42", "random_state = 43")
    )
    assert not first.df.equals(changed.data_preparator.prepare_data().df)


def test_legacy_config_loads_lazily_once_and_honors_replacement(monkeypatch):
    # Load a fresh module to avoid polluting the configuration used by other tests.
    name = "assets._legacy_config_test"
    spec = importlib.util.spec_from_file_location(name, configuration.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    with monkeypatch.context() as context:
        context.setattr(
            configuration.configparser.ConfigParser,
            "read",
            lambda *a, **k: pytest.fail("import read config"),
        )
        spec.loader.exec_module(module)
    assert not any(field.name in vars(module) for field in fields(configuration.Config))
    with pytest.raises(AttributeError):
        _ = module.unknown_option
    calls = []
    original = module.load_config

    def load():
        calls.append(True)
        return original()

    monkeypatch.setattr(module, "load_config", load)
    explicit = module.load_config()
    assert "imputers" not in vars(module)
    first_imputers = module.imputers
    assert first_imputers
    assert module.imputers is first_imputers
    assert len(calls) == 2  # explicit call and one lazy load
    assert all(field.name in vars(module) for field in fields(configuration.Config))
    assert module.get_field(fields={"present": 3}, field_name="present") == 3
    assert module.get_field({}, "missing") is None
    assert module.mar_chunk == {}
    replacement = DataGenerator(1, 10, NoAdditionalMissingValues(), random_state=43)
    module.__dict__["data_preparator"] = replacement
    prepared, actual = module.prepare_data_from_config()
    assert actual is replacement.actual_values
    pd.testing.assert_frame_equal(prepared.df, replacement.prepare_data().df)
    assert explicit.data_preparator is not replacement
    assert len(calls) == 2


def test_legacy_lazy_load_preserves_early_overrides(monkeypatch):
    name = "assets._early_legacy_config_test"
    spec = importlib.util.spec_from_file_location(name, configuration.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    replacement = DataGenerator(1, 10, NoAdditionalMissingValues(), random_state=42)
    module.__dict__["data_preparator"] = replacement
    module.__dict__["plot_results"] = False
    assert module.imputers
    assert module.data_preparator is replacement
    assert module.plot_results is False
    prepared, _actual = module.prepare_data_from_config()
    assert len(prepared.df) == 10


@pytest.mark.parametrize("complete", [False, True])
def test_optional_legacy_field_skips_load_for_complete_overrides(monkeypatch, complete):
    name = "assets._optional_legacy_config_test"
    spec = importlib.util.spec_from_file_location(name, configuration.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    for field in fields(module.Config):
        if field.name == "execution_random_state":
            continue
        if not complete and field.name == "missing_values":
            continue
        module.__dict__[field.name] = object()

    def load_config():
        raise AssertionError("optional field read configuration")

    monkeypatch.setattr(module, "load_config", load_config)
    if complete:
        assert module.execution_random_state is None
        assert module.__dict__["execution_random_state"] is None
    else:
        with pytest.raises(AssertionError, match="read configuration"):
            _ = module.execution_random_state


BASE_CONF_TEMPLATE = """
[imputation_algorithms]
imputation_algorithms = {imputers}

[plot]
plot_results = False

[profile]
profile = False

[dataset_selection]
used_dataset = gen

[gen]
type = generated
sample_size = 100
linear_interpolation_ratio = 1
missing_rate = 0.3
missing_type = MCAR
"""

SHIPPED_HYPERPARAMETERS = Path("assets/hyperparameters.conf")

# Minimal hyperparameters content for tests that exercise the config
# machinery itself rather than the shipped hyperparameters.
MINIMAL_HYPERPARAMETERS = """
[KnnSampler]
"""


def write_conf(tmp_path: Path, content: str, name: str = "config.conf") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def load_from_conf(
    tmp_path: Path,
    conf: str,
    hyperparameters: str | None = None,
    hyperparameters_path: Path | None = None,
) -> configuration.Config:
    """Load a Config from a written config file, using minimal
    hyperparameters by default."""
    return configuration.load_config(
        config_path=write_conf(tmp_path, conf),
        hyperparameters_path=(
            hyperparameters_path
            if hyperparameters_path is not None
            else write_conf(
                tmp_path,
                hyperparameters
                if hyperparameters is not None
                else MINIMAL_HYPERPARAMETERS,
                "hyperparameters.conf",
            )
        ),
    )


def base_conf(imputers: str = "KnnSampler", extra: str = "") -> str:
    return BASE_CONF_TEMPLATE.format(imputers=imputers) + extra


def profile_conf(dump_path_line: str, imputers: str = "KnnSampler") -> str:
    return f"""
[imputation_algorithms]
imputation_algorithms = {imputers}

[plot]
plot_results = False

[profile]
profile = True
{dump_path_line}
print_stats = False

[dataset_selection]
used_dataset = gen

[gen]
type = generated
sample_size = 100
linear_interpolation_ratio = 1
missing_rate = 0.3
missing_type = MCAR
"""


class TestLoadConfigDefaults:
    def test_load_default_config_structural(self):
        """Structural invariants only: config.conf is user-editable."""
        config = configuration.load_config()
        assert isinstance(config, configuration.Config)
        assert config.imputers
        assert set(config.imputers) <= KNOWN_IMPUTERS
        assert isinstance(config.data_preparator, DataPreparator)
        assert isinstance(config.missing_values, (Rate, int))
        assert isinstance(config.plot_results, bool)
        assert isinstance(config.profile, bool)
        assert config.profile_dump_path is None or isinstance(
            config.profile_dump_path, Path
        )
        assert config.execution_random_state is None or isinstance(
            config.execution_random_state, int
        )


class TestLoadConfigErrors:
    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config file not found"):
            configuration.load_config(
                config_path=tmp_path / "does_not_exist.conf",
                hyperparameters_path=SHIPPED_HYPERPARAMETERS,
            )

    def test_missing_hyperparameters_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="hyperparameters file"):
            configuration.load_config(
                config_path=write_conf(tmp_path, base_conf()),
                hyperparameters_path=tmp_path / "does_not_exist.conf",
            )

    def test_unknown_imputer_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unrecognized imputer"):
            load_from_conf(tmp_path, base_conf(imputers="NotAnImputer"))

    def test_unknown_dataset_section_raises(self, tmp_path):
        conf = base_conf().replace("used_dataset = gen", "used_dataset = nosuch")
        with pytest.raises(ValueError, match="dataset section 'nosuch'"):
            load_from_conf(tmp_path, conf)

    def test_missing_rate_and_values_raises(self, tmp_path):
        conf = base_conf().replace("missing_rate = 0.3\n", "")
        with pytest.raises(ValueError, match="missing_rate"):
            load_from_conf(tmp_path, conf)

    def test_invalid_missing_type_raises(self, tmp_path):
        conf = base_conf().replace("missing_type = MCAR", "missing_type = MNAR")
        with pytest.raises(ValueError, match="missing type is not defined"):
            load_from_conf(tmp_path, conf)

    def test_invalid_sample_size_raises(self, tmp_path):
        conf = base_conf().replace("sample_size = 100", "sample_size = -5")
        with pytest.raises(ValueError, match="sample_size"):
            load_from_conf(tmp_path, conf)
        conf = base_conf().replace("sample_size = 100", "sample_size = abc")
        with pytest.raises(ValueError, match="sample_size"):
            load_from_conf(tmp_path, conf)

    def test_generated_without_sample_size_raises(self, tmp_path):
        """A blank sample_size must parse to None and be rejected for generated data."""
        conf = base_conf().replace("sample_size = 100", "sample_size =")
        with pytest.raises(ValueError, match="sample_size is not defined"):
            load_from_conf(tmp_path, conf)

    def test_unknown_dataset_type_raises(self, tmp_path):
        conf = base_conf().replace("type = generated", "type = json")
        with pytest.raises(ValueError, match=r"<generated \| excel>"):
            load_from_conf(tmp_path, conf)

    def test_mar_chunk_fields_required(self, tmp_path):
        # base_conf's [gen] section defines no MAR_chunk_start/MAR_chunk_end
        conf = base_conf().replace("missing_type = MCAR", "missing_type = MAR")
        with pytest.raises(ValueError, match="MAR_chunk_start"):
            load_from_conf(tmp_path, conf)


class TestDumpPath:
    def test_quoted_nonexistent_dump_path_is_kept(self, tmp_path):
        """Regression: a quoted path to a file that does not exist yet must be
        kept as a Path (dump_stats creates it), not silently dropped."""
        conf = profile_conf('dump_path = "some/new/dir/profiling_results.pkl"')
        config = load_from_conf(tmp_path, conf)
        assert config.profile_dump_path is not None
        assert not config.profile_dump_path.exists()
        assert config.profile_dump_path.name == "profiling_results.pkl"

    def test_relative_dump_path_anchored_to_project_root(self, tmp_path):
        conf = profile_conf("dump_path = assets/profiling/out.pkl")
        config = load_from_conf(tmp_path, conf)
        assert (
            config.profile_dump_path
            == configuration.PROJECT_ROOT / "assets/profiling/out.pkl"
        )

    @pytest.mark.parametrize("raw", ["", "   ", '""', "''", '"   "', "'   '"])
    def test_empty_dump_path_is_none(self, tmp_path, raw):
        conf = profile_conf(f"dump_path = {raw}")
        config = load_from_conf(tmp_path, conf)
        assert config.profile_dump_path is None

    @pytest.mark.parametrize(
        "raw, filename",
        [
            ('"  stats.pkl  "', "  stats.pkl  "),
            ("'\"stats.pkl\"'", '"stats.pkl"'),
        ],
    )
    def test_dump_path_preserves_quoted_filename_characters(
        self, tmp_path, raw, filename
    ):
        config = load_from_conf(tmp_path, profile_conf(f"dump_path = {raw}"))
        assert config.profile_dump_path == configuration.PROJECT_ROOT / filename

    def test_omitted_dump_path_is_none(self, tmp_path):
        conf = profile_conf("")
        config = load_from_conf(tmp_path, conf)
        assert config.profile_dump_path is None


class TestHyperparameters:
    def test_percentile_mirroring_lower_only(self, tmp_path):
        hyperparameters = """
[KnnSampler]
lower_percentiles = 5, 10 : int[]
compute_bounds = True : bool
"""
        config = load_from_conf(tmp_path, base_conf(), hyperparameters)
        assert config.imputers[KnnSampler]["upper_percentiles"] == [95, 90]

    def test_percentile_mirroring_upper_only(self, tmp_path):
        hyperparameters = """
[KnnSampler]
upper_percentiles = 95, 90 : int[]
compute_bounds = True : bool
"""
        config = load_from_conf(tmp_path, base_conf(), hyperparameters)
        assert config.imputers[KnnSampler]["lower_percentiles"] == [5, 10]

    def test_missing_colon_format_raises(self, tmp_path):
        hyperparameters = """
[KnnSampler]
n_neighbors = 5
"""
        with pytest.raises(ValueError, match="Format error"):
            load_from_conf(tmp_path, base_conf(), hyperparameters)

    def test_unknown_type_raises(self, tmp_path):
        hyperparameters = """
[KnnSampler]
n_neighbors = 5 : decimal
"""
        with pytest.raises(ValueError, match="Unknown type"):
            load_from_conf(tmp_path, base_conf(), hyperparameters)

    def test_conversion_failure_raises(self, tmp_path):
        hyperparameters = """
[KnnSampler]
n_neighbors = abc : int
"""
        with pytest.raises(ValueError, match="Could not convert"):
            load_from_conf(tmp_path, base_conf(), hyperparameters)

    def test_ml_data_option_forbidden(self, tmp_path):
        hyperparameters = """
[KnnSampler]
ml_data = 1 : int
"""
        with pytest.raises(ValueError, match="ml_data is not configurable"):
            load_from_conf(tmp_path, base_conf(), hyperparameters)

    def test_unused_section_warns(self, tmp_path):
        hyperparameters = """
[Bogus]
n_neighbors = 5 : int
"""
        with pytest.warns(UserWarning, match="Bogus"):
            load_from_conf(tmp_path, base_conf(), hyperparameters)


class TestExecutionSeed:
    def test_execution_random_state_is_parsed(self, tmp_path):
        conf = base_conf(extra="\n[execution]\nexecution_random_state = 123")
        config = load_from_conf(tmp_path, conf)
        assert config.execution_random_state == 123

    @pytest.mark.parametrize(
        "extra",
        ["", "\n[execution]\nexecution_random_state ="],
    )
    def test_missing_or_blank_execution_seed_is_none(self, tmp_path, extra):
        config = load_from_conf(tmp_path, base_conf(extra=extra))
        assert config.execution_random_state is None

    def test_seed_zero_is_valid(self, tmp_path):
        conf = base_conf(extra="\n[execution]\nexecution_random_state = 0")
        assert load_from_conf(tmp_path, conf).execution_random_state == 0

    @pytest.mark.parametrize("raw", ["-1", "abc"])
    def test_invalid_execution_seed_raises(self, tmp_path, raw):
        conf = base_conf(extra=f"\n[execution]\nexecution_random_state = {raw}")
        with pytest.raises(
            ValueError,
            match=rf"execution_random_state.*non-negative integer.*{re.escape(raw)}"
            ".*execution configuration",
        ):
            load_from_conf(tmp_path, conf)

    def test_execution_seed_does_not_change_dataset_seeds(self, tmp_path):
        conf = base_conf(
            extra="random_state = 7\n[execution]\nexecution_random_state = 42"
        )
        config = load_from_conf(tmp_path, conf)
        preparation_seed, missingness_seed = split_random_state(7)

        assert config.execution_random_state == 42
        assert config.data_preparator.random_state == preparation_seed
        assert isinstance(config.data_preparator.missing_generator, (Mcar, Mar))
        assert config.data_preparator.missing_generator.random_state == missingness_seed


class TestDatasetFields:
    @pytest.mark.parametrize("seed", [0, 42, 2**128])
    def test_random_state_plumbed_to_preparator(self, tmp_path, seed):
        conf = base_conf(extra=f"random_state = {seed}")
        config = load_from_conf(tmp_path, conf)
        assert isinstance(config.data_preparator, DataGenerator)
        preparation_seed, missingness_seed = split_random_state(seed)
        assert config.data_preparator.random_state == preparation_seed
        assert isinstance(config.data_preparator.missing_generator, Mcar)
        assert config.data_preparator.missing_generator.random_state == missingness_seed
        assert preparation_seed != missingness_seed

    @pytest.mark.parametrize("option", ["", "random_state ="])
    def test_blank_random_state_is_none(self, tmp_path, option):
        conf = base_conf(extra=option)
        config = load_from_conf(tmp_path, conf)
        assert config.data_preparator.random_state is None
        assert isinstance(config.data_preparator.missing_generator, Mcar)
        assert config.data_preparator.missing_generator.random_state is None

    @pytest.mark.parametrize("raw", ["-5", "abc"])
    def test_invalid_random_state_raises(self, tmp_path, raw):
        conf = base_conf(extra=f"random_state = {raw}")
        with pytest.raises(
            ValueError,
            match=rf"random_state.*non-negative integer.*{re.escape(raw)}.*dataset gen\b",
        ):
            load_from_conf(tmp_path, conf)

    def test_excel_path_anchored_to_project_root(self, tmp_path):
        conf = base_conf().replace(
            """
[gen]
type = generated
sample_size = 100
linear_interpolation_ratio = 1
missing_rate = 0.3
missing_type = MCAR
""",
            """
[gen]
type = excel
path = "assets/datasets/publishable_data.xlsx"
sheet = Sheet1
input = IRRADIATION
target = DC_POWER
missing_rate = 0.3
missing_type = MCAR
""",
        )
        config = load_from_conf(tmp_path, conf)
        assert config.data_preparator.dataset_descriptor.path == str(
            configuration.PROJECT_ROOT / "assets/datasets/publishable_data.xlsx"
        )
