"""Tests for regime filters and walk-forward utilities."""

import pandas as pd
import pytest

from src.data import load_panel
from src.regime import trend_filter, vol_filter, combined_filter
from src.backtest import walk_forward_segments, run_backtest
from src.signals import momentum_total_return


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return load_panel()


def test_trend_filter_is_boolean_and_aligned(panel):
    f = trend_filter(panel, market_ticker="spy", lookback_days=200)
    # Should be a Series with datetime index
    assert isinstance(f, pd.Series)
    assert pd.api.types.is_datetime64_any_dtype(f.index)
    # First 200 days are NaN (insufficient history); after that should be bool-like
    valid = f.dropna()
    assert valid.dtype == bool
    # Should be mostly True over a long bull-tilted sample (>50%)
    assert valid.mean() > 0.5


def test_vol_filter_runs(panel):
    f = vol_filter(panel, market_ticker="spy", vol_window=21, expanding_pct_threshold=0.8)
    assert isinstance(f, pd.Series)
    # Should produce a mix of True/False after warmup
    valid = f.iloc[100:]
    assert valid.any() and (~valid).any()


def test_combined_filter_is_and_of_components(panel):
    only_trend = combined_filter(panel, use_trend=True, use_vol=False)
    only_vol = combined_filter(panel, use_trend=False, use_vol=True)
    both = combined_filter(panel, use_trend=True, use_vol=True)
    # Combined must be no looser than either individual
    common = both.index.intersection(only_trend.index).intersection(only_vol.index)
    assert (both.loc[common] <= only_trend.loc[common]).all()
    assert (both.loc[common] <= only_vol.loc[common]).all()


def test_walk_forward_segments_partition_correctly():
    dates = pd.date_range("2014-01-01", "2026-01-01", freq="B")
    segs = walk_forward_segments(dates, train_years=4.0, test_years=1.5, step_years=1.5)
    assert len(segs) >= 4
    # Each test window starts immediately after the corresponding train window ends
    for train_s, train_e, test_s, test_e in segs:
        assert test_s > train_e
        assert (test_s - train_e).days <= 2
        assert test_e > test_s


def test_regime_gate_reduces_invested_time(panel):
    panel = panel.copy()
    panel["mom6"] = momentum_total_return(panel, 6, 1)
    gate = trend_filter(panel, "spy", 200)
    gated = run_backtest(panel, "mom6", n_long=10, rebalance="ME", regime_gate=gate)
    ungated = run_backtest(panel, "mom6", n_long=10, rebalance="ME")
    # With a gate the invested fraction must be <= 1.0; ungated is always 1.0 when sufficient signal
    assert gated.invested_pct is not None
    assert gated.invested_pct < 1.0
    assert ungated.invested_pct == 1.0 or ungated.invested_pct is None
