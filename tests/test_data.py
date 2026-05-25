"""Tests for data integrity and loading."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_panel, load_universe, load_wide_closes, to_wide


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return load_panel()


def test_panel_loads_and_has_expected_shape(panel):
    assert len(panel) > 100_000, "panel should have >100k row-observations"
    expected_cols = {
        "date", "ticker", "category", "close", "return",
        "log_return", "vol_10d_ann", "signal_ret_over_vol", "sheet_name",
    }
    assert expected_cols.issubset(panel.columns)


def test_panel_has_43_tickers(panel):
    assert panel["ticker"].nunique() == 43


def test_panel_has_no_duplicate_keys(panel):
    dups = panel.duplicated(subset=["date", "ticker"]).sum()
    assert dups == 0, f"found {dups} duplicate (date, ticker) rows"


def test_panel_prices_are_positive(panel):
    assert (panel["close"] > 0).all(), "all closing prices must be positive"


def test_universe_loads(panel):
    uni = load_universe()
    assert len(uni) == 43
    assert set(uni["ticker"]) == set(panel["ticker"].unique())


def test_no_unadjusted_splits(panel):
    """After cleaning, no single-day return should exceed 30% in magnitude
    for any ETF in this universe — real ETFs simply don't move that much.
    """
    extreme = panel[panel["return"].abs() > 0.30]
    assert len(extreme) == 0, (
        f"unexpected extreme daily returns (possible unadjusted split):\n{extreme}"
    )


def test_to_wide_roundtrip(panel):
    wide = to_wide(panel, "close")
    assert wide.shape[1] == 43
    # Spot-check a known SPY value (2014-11-19 close = 205.22 in the source workbook)
    assert np.isclose(wide.loc["2014-11-19", "spy"], 205.22, rtol=1e-4)


def test_wide_closes_matches_pivot(panel):
    wide_from_csv = load_wide_closes()
    wide_pivot = to_wide(panel, "close")
    common = wide_from_csv.index.intersection(wide_pivot.index)
    left = wide_from_csv.loc[common].sort_index(axis=1)
    right = wide_pivot.loc[common].sort_index(axis=1)
    # Normalize column-index name (CSV roundtrip drops it)
    left.columns.name = right.columns.name = None
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-6)
