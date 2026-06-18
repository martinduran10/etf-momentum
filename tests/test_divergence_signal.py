"""Tests for the reproducible divergence signal (Phase 3 input).

These pin the spread and percentile mechanics on synthetic data and prove the
no-look-ahead property: a date's percentile depends only on spreads up to and
including that date, so appending future data never changes an earlier value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.divergence_signal import (
    compute_divergence_percentile,
    compute_divergence_spread,
)


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


# --------------------------------------------------------------------------- #
# Spread: top-5 sum minus bottom-5 sum
# --------------------------------------------------------------------------- #


def test_spread_is_top5_sum_minus_bottom5_sum() -> None:
    cols = [f"e{i}" for i in range(12)]
    vals = [12.0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    slow = pd.DataFrame([vals], index=_days(1), columns=cols)
    # top5 = 12+11+10+9+8 = 50 ; bottom5 = 1+2+3+4+5 = 15 ; spread = 35
    assert compute_divergence_spread(slow).iloc[0] == 35.0


def test_spread_nan_when_fewer_than_ten_valid() -> None:
    cols = [f"e{i}" for i in range(12)]
    row = [1.0, 2, 3, 4, 5, 6, 7, 8, 9] + [np.nan] * 3  # only 9 valid
    slow = pd.DataFrame([row], index=_days(1), columns=cols)
    assert np.isnan(compute_divergence_spread(slow).iloc[0])


# --------------------------------------------------------------------------- #
# Percentile: expanding, inclusive, past-only
# --------------------------------------------------------------------------- #


def test_percentile_is_expanding_and_inclusive() -> None:
    spread = pd.Series([10.0, 5.0, 20.0, 5.0], index=_days(4))
    pct = compute_divergence_percentile(spread)
    # day0: 1/1 ; day1: 5 is min of {10,5} -> 1/2 ; day2: 20 is max -> 3/3 ;
    # day3: values <=5 in {10,5,20,5} are {5,5} -> 2/4
    assert pct.tolist() == [1.0, 0.5, 1.0, 0.5]


def test_percentile_bounds() -> None:
    rng = np.random.default_rng(1)
    spread = pd.Series(rng.normal(0, 1, 200), index=_days(200))
    pct = compute_divergence_percentile(spread).dropna()
    assert pct.min() > 0.0 and pct.max() <= 1.0


def test_nan_spreads_are_skipped_in_ranking() -> None:
    spread = pd.Series([np.nan, 10.0, np.nan, 5.0], index=_days(4))
    pct = compute_divergence_percentile(spread)
    assert np.isnan(pct.iloc[0]) and np.isnan(pct.iloc[2])
    assert pct.iloc[1] == 1.0  # first valid -> 1/1
    assert pct.iloc[3] == 0.5  # 5 <= {10, 5} -> 1/2


def test_no_look_ahead_truncation_invariance() -> None:
    """A date's percentile uses only spreads up to that date: truncating the
    series must leave every earlier percentile unchanged."""
    rng = np.random.default_rng(2)
    spread = pd.Series(rng.normal(0, 1, 120), index=_days(120))
    full = compute_divergence_percentile(spread)
    for k in (30, 60, 90):
        trunc = compute_divergence_percentile(spread.iloc[:k])
        pd.testing.assert_series_equal(full.iloc[:k], trunc, check_freq=False)
