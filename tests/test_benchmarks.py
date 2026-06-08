"""Tests for the SPY buy-and-hold benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.benchmarks import (
    benchmark_summary,
    compounded_equity_curve,
    compounded_max_drawdown,
    spy_buy_and_hold_returns,
)
from src.data import load_closes
from src.stack_backtest import HEADLINE_START, run_stack_portfolio


def _synthetic_returns(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, size=n), index=dates)


def test_spy_aligns_with_strategy_length_and_start() -> None:
    closes = load_closes()
    strategy_returns = run_stack_portfolio(closes)["headline_returns"]
    spy_returns = spy_buy_and_hold_returns(closes)

    assert len(spy_returns) == len(strategy_returns)
    assert spy_returns.index[0] == pd.Timestamp(HEADLINE_START)
    assert spy_returns.index.equals(strategy_returns.index)


def test_compounded_equity_curve_matches_closed_form() -> None:
    rets = _synthetic_returns()
    curve = compounded_equity_curve(rets)
    expected = (1.0 + rets).cumprod() - 1.0
    pd.testing.assert_series_equal(curve, expected, check_names=False)


def test_compounded_max_drawdown_matches_wealth_formula() -> None:
    # Hand-built series with a clear peak and trough.
    rets = pd.Series(
        [0.10, 0.10, -0.20, -0.10, 0.05],
        index=pd.date_range("2020-01-01", periods=5, freq="B"),
    )
    wealth = (1.0 + rets).cumprod()
    expected = float((wealth / wealth.cummax() - 1.0).min())

    assert compounded_max_drawdown(rets) == expected


def test_benchmark_summary_keys_and_internal_consistency() -> None:
    rets = _synthetic_returns(n=252, seed=11)
    metrics = benchmark_summary(rets)

    expected_keys = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    }
    assert set(metrics) == expected_keys

    # total_return is the compounded growth - 1.
    assert np.isclose(metrics["total_return"], (1.0 + rets).prod() - 1.0)

    # sharpe_ratio = annualized_return / annualized_volatility (rf=0).
    assert np.isclose(
        metrics["sharpe_ratio"],
        metrics["annualized_return"] / metrics["annualized_volatility"],
    )
