"""Tests for the risk-adjusted return and slow-signal computations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import (
    compute_log_returns,
    compute_vol_10,
    load_closes,
)
from src.signals import compute_risk_adj_return, compute_slow_signal


def _toy_closes() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    prices = 100 * np.cumprod(1 + 0.01 * np.sin(np.arange(30)))
    return pd.DataFrame({"xyz": prices}, index=dates)


def test_risk_adj_return_matches_manual_formula() -> None:
    closes = _toy_closes()
    log_ret = compute_log_returns(closes)
    vol_10 = compute_vol_10(log_ret)
    expected = log_ret / vol_10.replace(0.0, np.nan)
    rar = compute_risk_adj_return(closes)
    pd.testing.assert_frame_equal(rar, expected)


def test_log_return_definition() -> None:
    closes = _toy_closes()
    log_ret = compute_log_returns(closes)
    manual = np.log(closes["xyz"].iloc[5] / closes["xyz"].iloc[4])
    assert log_ret["xyz"].iloc[5] == manual


def test_vol_10_uses_sample_std_and_annualizes() -> None:
    closes = _toy_closes()
    log_ret = compute_log_returns(closes)
    vol = compute_vol_10(log_ret)
    window = log_ret["xyz"].iloc[6:16]  # ten observations ending at idx 15
    manual = window.std(ddof=1) * np.sqrt(252)
    assert np.isclose(vol["xyz"].iloc[15], manual)


def test_early_values_are_nan() -> None:
    closes = _toy_closes()
    rar = compute_risk_adj_return(closes)
    # vol_10 needs 10 returns -> first valid risk_adj_return is at index 10.
    assert rar["xyz"].iloc[:10].isna().all()
    assert not np.isnan(rar["xyz"].iloc[10])


def test_slow_signal_is_rolling_mean() -> None:
    closes = load_closes()
    rar = compute_risk_adj_return(closes)
    slow = compute_slow_signal(rar, lookback=260)
    col = closes.columns[0]
    manual = rar[col].iloc[400 - 260 + 1 : 400 + 1].mean()
    assert np.isclose(slow[col].iloc[400], manual)


def test_slow_signal_nan_before_window_fills() -> None:
    closes = load_closes()
    rar = compute_risk_adj_return(closes)
    slow = compute_slow_signal(rar, lookback=260)
    col = closes.columns[0]
    # risk_adj_return starts at row 10; a 260-value window mean is first
    # defined once 260 valid values exist, i.e. at row 269.
    assert np.isnan(slow[col].iloc[268])
    assert not np.isnan(slow[col].iloc[269])
