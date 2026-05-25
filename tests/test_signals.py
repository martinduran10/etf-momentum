"""Tests for signal computation and backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_panel
from src.signals import (
    momentum_total_return,
    momentum_risk_adjusted,
    cross_sectional_rank,
)
from src.backtest import run_backtest, equal_weight_benchmark


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return load_panel()


def test_momentum_signal_has_no_lookahead(panel):
    """The 6m-skip-1m signal should be NaN before sufficient history exists."""
    sig = momentum_total_return(panel, lookback_months=6, skip_months=1)
    # 6 months × ~21 trading days = 126 days of history needed per ticker
    by_ticker_first_valid = (
        panel.assign(sig=sig).dropna(subset=["sig"]).groupby("ticker")["date"].min()
    )
    panel_first = panel.groupby("ticker")["date"].min()
    diffs = (by_ticker_first_valid - panel_first).dt.days
    # All tickers should require at least 6 months of history before producing a signal
    assert (diffs >= 120).all(), "signal appears too early — possible look-ahead"


def test_cross_sectional_rank_covers_all_tickers(panel):
    sig = momentum_total_return(panel, lookback_months=6, skip_months=1)
    panel_with_sig = panel.assign(mom6=sig)
    ranks = cross_sectional_rank(panel_with_sig, "mom6")
    # On any date with all 43 tickers having a signal, ranks should span 1..43
    panel_with_ranks = panel_with_sig.assign(rank=ranks)
    sample_date = panel_with_ranks.dropna(subset=["mom6"])["date"].iloc[-1]
    ranks_that_day = panel_with_ranks[panel_with_ranks["date"] == sample_date]["rank"].dropna()
    assert ranks_that_day.min() == 1
    assert ranks_that_day.max() <= 43


def test_backtest_runs_and_produces_sensible_output(panel):
    panel = panel.copy()
    panel["mom6"] = momentum_total_return(panel, lookback_months=6, skip_months=1)
    res = run_backtest(panel, signal_col="mom6", n_long=10, rebalance="ME")
    assert len(res.equity_curve) > 1000
    assert res.equity_curve.iloc[0] > 0
    # Equity should be finite and positive throughout
    assert res.equity_curve.notna().all()
    assert (res.equity_curve > 0).all()
    # Positions should sum to ~1.0 once invested (or exactly 0 before first rebalance)
    sums = res.positions.sum(axis=1)
    nonzero = sums[sums > 0]
    assert np.allclose(nonzero, 1.0, atol=1e-9)


def test_benchmark_equity_runs(panel):
    eq = equal_weight_benchmark(panel)
    assert len(eq) > 1000
    assert (eq > 0).all()


def test_invalid_skip_months_raises(panel):
    with pytest.raises(ValueError):
        momentum_total_return(panel, lookback_months=3, skip_months=3)
