"""Exercise profiling output paths using the real optional writer."""

import pytest

from assets.profiling.line_profile import Profiler
from tests.config_test import load_from_conf, profile_conf


@pytest.mark.parametrize("as_string", [False, True])
def test_profile_dump_creates_parent_from_config(tmp_path, as_string):
    pytest.importorskip("line_profiler")
    destination = tmp_path / "new" / "nested" / "stats.pkl"
    config = load_from_conf(tmp_path, profile_conf(f'dump_path = "{destination}"'))
    assert config.profile_dump_path is not None
    assert not destination.parent.exists()
    profiler = Profiler([])
    assert profiler.line_profiler is not None
    assert profiler.start_profile(lambda: 7) == 7
    path = str(config.profile_dump_path) if as_string else config.profile_dump_path
    assert profiler.dump_stats(path) is profiler
    assert destination.stat().st_size > 0
    profiler.dump_stats(path)
    blocked_parent = tmp_path / "not_a_directory"
    blocked_parent.write_text("file", encoding="utf-8")
    with pytest.raises(FileExistsError):
        profiler.dump_stats(blocked_parent / "stats.pkl")
    absent = tmp_path / "disabled" / "stats.pkl"
    profiler.line_profiler = None
    assert profiler.dump_stats(str(absent) if as_string else absent) is profiler
    assert not absent.parent.exists()
