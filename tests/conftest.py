"""Shared headless plotting setup and dataset fixtures."""

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")


@pytest.fixture
def multi_sheet_workbook(tmp_path: Path) -> Path:
    """Two identically shaped sheets with distinguishable target values."""
    path = tmp_path / "two_sheets.xlsx"
    with pd.ExcelWriter(path) as writer:
        for sheet, offset in (("first", -100), ("selected", 100)):
            pd.DataFrame(
                {
                    "input": [1, 1, 2, 2, 3, 3, 4, 4],
                    "target": [offset + i for i in range(8)],
                    "source": [sheet] * 8,
                }
            ).to_excel(writer, sheet_name=sheet, index=False)
    return path
