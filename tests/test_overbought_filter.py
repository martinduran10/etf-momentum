"""Tests for the Phase 2 overbought-market cash filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.overbought_filter import (
    CASH_LAG_RANGE,
    apply_filter,
    build_cash_mask,
    load_overbought_signal,
)


def _toy_trading_days(n: int = 30) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_isolated_zero_flags_t3_through_t6() -> None:
    """A single 0-signal at T marks exactly T+3..T+6 as cash."""
    trading_days = _toy_trading_days()
    signal = pd.Series(1, index=trading_days)
    signal.iloc[5] = 0

    mask = build_cash_mask(signal, trading_days)

    expected = pd.Series(False, index=trading_days)
    expected.iloc[8:12] = True  # positions 5+3, 5+4, 5+5, 5+6
    pd.testing.assert_series_equal(mask, expected, check_names=False)


def test_overlapping_zeros_merge_into_continuous_block() -> None:
    """A run of consecutive 0s produces a continuous cash block."""
    trading_days = _toy_trading_days()
    signal = pd.Series(1, index=trading_days)
    signal.iloc[5:8] = 0  # zeros at 5, 6, 7

    mask = build_cash_mask(signal, trading_days)

    # Cash from first zero's T+3 (5+3=8) to last zero's T+6 (7+6=13), inclusive.
    expected = pd.Series(False, index=trading_days)
    expected.iloc[8:14] = True
    pd.testing.assert_series_equal(mask, expected, check_names=False)


def test_no_lookahead_mask_independent_of_recent_signals() -> None:
    """mask[t] depends only on signal at t-3 and earlier."""
    trading_days = _toy_trading_days()
    base = pd.Series(1, index=trading_days)
    base.iloc[5] = 0
    mask_base = build_cash_mask(base, trading_days)

    altered = base.copy()
    # Flipping signals at t-2, t-1, t for t=10 must not change mask[10].
    altered.iloc[8] = 0
    altered.iloc[9] = 0
    altered.iloc[10] = 0
    mask_altered = build_cash_mask(altered, trading_days)

    assert mask_base.iloc[10] == mask_altered.iloc[10]


def test_lookback_predating_filter_treated_as_ok() -> None:
    """Trading days whose lookbacks predate the filter are not forced to cash."""
    trading_days = _toy_trading_days()
    # Filter contains only a single 1-signal far into the calendar.
    signal = pd.Series([1], index=[trading_days[15]])

    mask = build_cash_mask(signal, trading_days)

    # With no 0s anywhere in the filter (and missing days treated as 1),
    # no day should be cash.
    assert not mask.any()


def test_strategy_day_missing_from_filter_treated_as_ok() -> None:
    """A trading day not present in the filter contributes a 1 (OK)."""
    trading_days = _toy_trading_days()
    # Filter omits positions 4..7 entirely; remaining values are all 1.
    keep = list(range(len(trading_days)))
    for i in (4, 5, 6, 7):
        keep.remove(i)
    signal = pd.Series(1, index=trading_days[keep])

    mask = build_cash_mask(signal, trading_days)

    # No zeros anywhere -> no cash days.
    assert not mask.any()


def test_apply_filter_zeros_cash_days_keeps_others() -> None:
    """Filtered return = 0 on cash days, = phase1 return otherwise."""
    idx = _toy_trading_days(5)
    phase1 = pd.Series([0.01, -0.02, 0.005, 0.0, -0.01], index=idx)
    mask = pd.Series([False, True, False, True, False], index=idx)

    filtered = apply_filter(phase1, mask)

    expected = pd.Series([0.01, 0.0, 0.005, 0.0, -0.01], index=idx)
    pd.testing.assert_series_equal(filtered, expected, check_names=False)


def test_apply_filter_handles_subset_mask() -> None:
    """A mask built over the full calendar can be applied to a subset slice."""
    full = _toy_trading_days(20)
    mask_full = pd.Series(False, index=full)
    mask_full.iloc[10:12] = True
    phase1 = pd.Series(np.arange(5, dtype=float) / 100.0, index=full[8:13])

    filtered = apply_filter(phase1, mask_full)

    # Positions 10 and 11 in the full index correspond to positions 2 and 3
    # in the slice (which spans positions 8..12).
    assert filtered.iloc[0] == phase1.iloc[0]
    assert filtered.iloc[1] == phase1.iloc[1]
    assert filtered.iloc[2] == 0.0
    assert filtered.iloc[3] == 0.0
    assert filtered.iloc[4] == phase1.iloc[4]


def test_load_overbought_signal_shape() -> None:
    """The shipped signal loads as a date-indexed 0/1 series."""
    sig = load_overbought_signal()
    assert isinstance(sig.index, pd.DatetimeIndex)
    assert sig.index.is_monotonic_increasing
    assert set(sig.unique()).issubset({0, 1})


def test_cash_lag_range_matches_spec() -> None:
    """The T+3..T+6 window is exactly four trading days wide."""
    assert list(CASH_LAG_RANGE) == [3, 4, 5, 6]
