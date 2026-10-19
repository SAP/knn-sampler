"""Check coverage aggregation and plots across mechanisms and missing rates."""

import matplotlib.pyplot as plt
import numpy as np

import benchmark_knnsampler as coverage
from src.data_preparation.data_preparation import Mar, Mcar, Rate


def coverage_rows():
    rows = []
    for geometry in coverage.geometries:
        for rate in (0.1, 0.3, 0.5):
            for iteration, probability in enumerate((0.8, 1.0)):
                for mechanism in (
                    Mcar(Rate(rate), random_state=iteration),
                    Mar(0.5, 1.5, Rate(rate), random_state=iteration),
                ):
                    for upper in coverage.upper_percentiles:
                        rows.append(
                            (
                                geometry,
                                mechanism,
                                upper,
                                100,
                                rate,
                                iteration,
                                probability,
                            )
                        )
    return rows


def test_coverage_grouping_and_plot_dimensions(monkeypatch):
    monkeypatch.setattr(coverage.plt, "show", lambda: None)
    groups = coverage.transform_results(coverage_rows())
    assert len(groups) == 4
    for _geometry, mechanism, means, stds in groups:
        assert isinstance(mechanism, (Mcar, Mar))
        for mean, std in zip(means, stds, strict=True):
            assert set(mean["Missing Rate"]) == {0.1, 0.3, 0.5}
            np.testing.assert_allclose(mean["Coverage Probability"], 0.9)
            np.testing.assert_allclose(
                std["Coverage Probability"], np.std([0.8, 1.0], ddof=1)
            )
    extra = coverage_rows() + [
        ("linear", Mar(-1, 1, Rate(0.3)), upper, 100, 0.3, iteration, probability)
        for upper in coverage.upper_percentiles
        for iteration, probability in enumerate((0.8, 1.0))
    ]
    more = coverage.transform_results(extra)
    assert len(more) == 5
    try:
        for batch in (groups[:1], groups, more):
            coverage.plot_all_in_one(batch, ["80", "90", "95"])
            assert len(plt.gcf().axes) == len(batch) * len(coverage.upper_percentiles)
            plt.close("all")
    finally:
        plt.close("all")
