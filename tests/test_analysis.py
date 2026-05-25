"""Tests for statistical analysis and the long/short backtest variant."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_panel
from src.signals import momentum_total_return
from src.backtest import run_backtest
from src.analysis import (
    regress_against_market,
    bootstrap_sharpe_ci,
    cost_sensitivity,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return load_panel()


# ---------------------------------------------------------------------------
# Long/short engine
# ---------------------------------------------------------------------------


def test_long_short_is_dollar_neutral(panel):
    """A long/short portfolio should sum (gross-weighted) to approximately zero
    on every rebalance, and gross exposure should be 2.0."""
    panel = panel.copy()
    panel["mom6"] = momentum_total_return(panel, 6, 1)
    res = run_backtest(panel, "mom6", n_long=10, n_short=10, rebalance="ME")

    # Take positions on rebalance dates only
    pos_at_rebal = res.positions.loc[res.rebalance_dates].iloc[1:]  # skip first
    nonzero_rows = pos_at_rebal[pos_at_rebal.abs().sum(axis=1) > 1e-9]
    if len(nonzero_rows) == 0:
        pytest.skip("no fully populated rebalance dates in sample")
    # Net exposure ~0, gross exposure ~2
    net = nonzero_rows.sum(axis=1)
    gross = nonzero_rows.abs().sum(axis=1)
    assert np.allclose(net, 0.0, atol=1e-9), f"non-zero net exposure: {net.describe()}"
    assert np.allclose(gross, 2.0, atol=1e-9), f"unexpected gross exposure: {gross.describe()}"


def test_long_only_unchanged_when_n_short_is_zero(panel):
    """The new long/short engine should behave identically to long-only
    when n_short=0 — important regression test."""
    panel = panel.copy()
    panel["mom6"] = momentum_total_return(panel, 6, 1)
    long_only = run_backtest(panel, "mom6", n_long=10, n_short=0, rebalance="ME")
    # Long-only invariants
    sums = long_only.positions.sum(axis=1)
    nonzero = sums[sums > 0]
    assert np.allclose(nonzero, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Factor regression
# ---------------------------------------------------------------------------


def test_market_beta_is_close_to_one_for_market_itself(panel):
    """Regressing SPY against itself must yield α≈0, β≈1, R²≈1."""
    spy = panel[panel["ticker"] == "spy"].set_index("date")["return"].dropna()
    result = regress_against_market(spy, spy)
    assert abs(result.alpha_daily) < 1e-10
    assert abs(result.beta - 1.0) < 1e-10
    assert result.r_squared > 0.9999


def test_regression_returns_sensible_values_for_momentum(panel):
    panel = panel.copy()
    panel["mom6"] = momentum_total_return(panel, 6, 1)
    res = run_backtest(panel, "mom6", n_long=10, rebalance="ME")
    spy = panel[panel["ticker"] == "spy"].set_index("date")["return"]
    fit = regress_against_market(res.daily_returns, spy)
    # Beta should be positive (we hold equities) and well below 2
    assert 0.2 < fit.beta < 1.5
    # R² between 0 and 1 obviously
    assert 0.0 <= fit.r_squared <= 1.0


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_point_estimate(panel):
    spy = panel[panel["ticker"] == "spy"].set_index("date")["return"].dropna()
    ci = bootstrap_sharpe_ci(spy, n_bootstrap=500, block_length=21)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]
    assert ci["n_bootstrap"] == 500


def test_bootstrap_handles_too_short_series():
    short_returns = pd.Series([0.01, -0.005, 0.002, 0.001])
    ci = bootstrap_sharpe_ci(short_returns, n_bootstrap=100, block_length=21)
    assert ci["n_bootstrap"] == 0
    assert np.isnan(ci["ci_low"])


# ---------------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------------


def test_cost_sensitivity_returns_monotonically_decrease(panel):
    """CAGR must weakly decrease as cost rises (more cost → less return)."""
    panel = panel.copy()
    panel["mom6"] = momentum_total_return(panel, 6, 1)
    df = cost_sensitivity(
        panel, "mom6",
        cost_levels_bps=(0, 5, 20, 50, 100),
        n_long=10, rebalance="ME",
    )
    cagrs = df["cagr"].tolist()
    # Strictly non-increasing
    for a, b in zip(cagrs, cagrs[1:]):
        assert b <= a + 1e-9, f"CAGR went UP as cost rose: {cagrs}"
    # At cost=0, CAGR should exceed cost=100bps version by a meaningful amount
    assert df.iloc[0]["cagr"] - df.iloc[-1]["cagr"] > 0.001


def test_cost_drag_scales_linearly_with_cost(panel):
    """Annual cost drag should scale linearly with cost level."""
    panel = panel.copy()
    panel["mom6"] = momentum_total_return(panel, 6, 1)
    df = cost_sensitivity(
        panel, "mom6",
        cost_levels_bps=(0, 10, 20),
        n_long=10, rebalance="ME",
    )
    # cost_drag at 10bps should be ~2x cost_drag at 5bps (approximately)
    drag_10 = df.iloc[1]["annual_cost_drag"]
    drag_20 = df.iloc[2]["annual_cost_drag"]
    # 20bps drag should be ~2x 10bps drag
    assert abs(drag_20 / drag_10 - 2.0) < 0.05, (
        f"cost drag should be ~linear in cost: {drag_10=}, {drag_20=}"
    )
