from pathlib import Path

import pytest

from assets.config import PROJECT_ROOT, Config, load_config
from src.data_preparation.data_preparation import (
    DataGenerator,
    DataPreparator,
    Mcar,
    Rate,
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
) -> Config:
    """Load a Config from a written config file, using minimal
    hyperparameters by default."""
    return load_config(
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
        config = load_config()
        assert isinstance(config, Config)
        assert config.imputers
        assert set(config.imputers) <= KNOWN_IMPUTERS
        assert isinstance(config.data_preparator, DataPreparator)
        assert isinstance(config.missing_values, (Rate, int))
        assert isinstance(config.plot_results, bool)
        assert isinstance(config.profile, bool)
        assert config.profile_dump_path is None or isinstance(
            config.profile_dump_path, Path
        )


class TestLoadConfigErrors:
    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config file not found"):
            load_config(
                config_path=tmp_path / "does_not_exist.conf",
                hyperparameters_path=SHIPPED_HYPERPARAMETERS,
            )

    def test_missing_hyperparameters_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="hyperparameters file"):
            load_config(
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
        assert config.profile_dump_path == PROJECT_ROOT / "assets/profiling/out.pkl"

    def test_empty_dump_path_is_none(self, tmp_path):
        conf = profile_conf("dump_path =")
        config = load_from_conf(tmp_path, conf)
        assert config.profile_dump_path is None

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


class TestDatasetFields:
    def test_random_state_plumbed_to_preparator(self, tmp_path):
        conf = base_conf(extra="random_state = 42")
        config = load_from_conf(tmp_path, conf)
        assert isinstance(config.data_preparator, DataGenerator)
        assert config.data_preparator.random_state == 42
        assert isinstance(config.data_preparator.missing_generator, Mcar)
        assert config.data_preparator.missing_generator.random_state == 42

    def test_blank_random_state_is_none(self, tmp_path):
        conf = base_conf(extra="random_state =")
        config = load_from_conf(tmp_path, conf)
        assert config.data_preparator.random_state is None

    def test_invalid_random_state_raises(self, tmp_path):
        conf = base_conf(extra="random_state = abc")
        with pytest.raises(ValueError, match="random_state"):
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
            PROJECT_ROOT / "assets/datasets/publishable_data.xlsx"
        )
