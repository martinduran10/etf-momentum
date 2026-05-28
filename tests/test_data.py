"""Data-integrity tests for the raw close-price panel."""

from __future__ import annotations

import pandas as pd

from src.data import load_closes, load_universe


def test_universe_size_is_43() -> None:
    universe = load_universe()
    assert len(universe) == 43
    assert universe["ticker"].nunique() == 43


def test_closes_cover_all_universe_tickers() -> None:
    closes = load_closes()
    universe = load_universe()
    expected = {t.lower() for t in universe["ticker"]}
    assert set(closes.columns) == expected
    assert closes.shape[1] == 43


def test_dates_sorted_unique_and_monotonic() -> None:
    closes = load_closes()
    assert isinstance(closes.index, pd.DatetimeIndex)
    assert closes.index.is_monotonic_increasing
    assert closes.index.is_unique


def test_no_negative_or_zero_prices() -> None:
    closes = load_closes()
    assert (closes > 0).all().all()


def test_no_missing_values_in_balanced_panel() -> None:
    closes = load_closes()
    # The data documentation promises a fully balanced panel with no gaps.
    assert not closes.isna().any().any()


def test_expected_date_range() -> None:
    closes = load_closes()
    assert closes.index[0] == pd.Timestamp("2014-07-03")
    assert closes.index[-1] == pd.Timestamp("2026-05-22")
    assert len(closes) == 2990
