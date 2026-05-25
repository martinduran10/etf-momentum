"""Cross-sectional momentum backtest engine.

Supports two portfolio variants:

* Long-only (default): equal-weighted top-N by signal.
* Long-short: equal-weighted top-N long and bottom-N short, dollar-neutral.

Either variant supports an optional regime gate that moves the strategy
to cash on rebalance dates where the gate is False.

All signal-to-execution lags are explicit: the signal at the rebalance date
is computed from data ending the day before, and positions earn returns from
the day after the rebalance onward. There is no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


RebalanceFreq = Literal["W", "ME", "QE"]


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex
    turnover: pd.Series
    config: dict = field(default_factory=dict)
    invested_pct: float | None = None


def run_backtest(
    panel: pd.DataFrame,
    signal_col: str,
    n_long: int = 10,
    n_short: int = 0,
    rebalance: RebalanceFreq = "ME",
    cost_bps_per_side: float = 5.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    regime_gate: pd.Series | None = None,
) -> BacktestResult:
    """Run a cross-sectional backtest.

    Args:
        panel: Long-format panel with ``date``, ``ticker``, ``return``,
            and the signal column.
        signal_col: Column to rank on (higher = better).
        n_long: Number of names to hold long.
        n_short: Number of names to hold short. ``0`` means long-only;
            otherwise the strategy is dollar-neutral with equal weights
            in each leg (each leg sums to ±1.0 / max(n_long, n_short)).
        rebalance: ``"W"``, ``"ME"`` (month-end) or ``"QE"`` (quarter-end).
        cost_bps_per_side: Per-side transaction cost in bps. Applied to
            the gross turnover (sum of |Δw|).
        start: Optional start date for the backtest window.
        end: Optional end date.
        regime_gate: Optional boolean Series indexed by date. When False on
            a rebalance date, the strategy holds zero positions for that
            period.

    Returns:
        A ``BacktestResult`` with equity curve, daily returns, positions,
        rebalance dates, turnover, config and invested fraction.
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    if start is not None:
        panel = panel[panel["date"] >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel["date"] <= pd.Timestamp(end)]

    returns = panel.pivot_table(index="date", columns="ticker", values="return").sort_index()
    signals = panel.pivot_table(index="date", columns="ticker", values=signal_col).sort_index()

    period_groups = returns.index.to_series().groupby(pd.Grouper(freq=rebalance))
    rebal_dates = pd.DatetimeIndex([g.iloc[-1] for _, g in period_groups if len(g)])
    first_signal = signals.dropna(how="all").index.min()
    rebal_dates = rebal_dates[rebal_dates >= first_signal]

    target_weights = pd.DataFrame(0.0, index=rebal_dates, columns=returns.columns)
    invested_flags = pd.Series(False, index=rebal_dates)

    long_short = n_short > 0

    for d in rebal_dates:
        if regime_gate is not None:
            gate_value = regime_gate.get(d)
            if gate_value is None or not bool(gate_value):
                continue
        s = signals.loc[d].dropna()
        if len(s) < (n_long + n_short):
            continue

        winners = s.nlargest(n_long).index
        target_weights.loc[d, winners] = 1.0 / n_long

        if long_short:
            losers = s.nsmallest(n_short).index
            target_weights.loc[d, losers] = -1.0 / n_short

        invested_flags[d] = True

    daily_weights = target_weights.reindex(returns.index).ffill().shift(1).fillna(0.0)
    strat_ret_gross = (daily_weights * returns).sum(axis=1)

    turnover = (target_weights.diff().abs().sum(axis=1)).fillna(
        target_weights.iloc[0].abs().sum()
    )
    cost_series = pd.Series(0.0, index=returns.index)
    cost_per_rebal = turnover * (cost_bps_per_side / 1e4)
    cost_series.loc[cost_per_rebal.index] = cost_per_rebal.values
    cost_series = cost_series.reindex(returns.index).fillna(0.0)

    strat_ret_net = strat_ret_gross - cost_series
    equity = (1.0 + strat_ret_net).cumprod()
    invested_pct = float(invested_flags.mean()) if len(invested_flags) else None

    return BacktestResult(
        equity_curve=equity,
        daily_returns=strat_ret_net,
        positions=daily_weights,
        rebalance_dates=rebal_dates,
        turnover=turnover,
        invested_pct=invested_pct,
        config={
            "signal_col": signal_col,
            "n_long": n_long,
            "n_short": n_short,
            "long_short": long_short,
            "rebalance": rebalance,
            "cost_bps_per_side": cost_bps_per_side,
            "start": str(start) if start else None,
            "end": str(end) if end else None,
            "regime_gated": regime_gate is not None,
        },
    )


def equal_weight_benchmark(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight benchmark across the full investable universe."""
    returns = panel.pivot_table(index="date", columns="ticker", values="return").sort_index()
    bench_ret = returns.mean(axis=1)
    return (1.0 + bench_ret.fillna(0.0)).cumprod()


def walk_forward_segments(
    dates: pd.DatetimeIndex,
    train_years: float = 5.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (train_start, train_end, test_start, test_end) segments."""
    dates = pd.DatetimeIndex(sorted(dates))
    if len(dates) == 0:
        return []
    start, end = dates[0], dates[-1]
    segments = []
    cursor = start
    one_year = pd.Timedelta(days=365)
    while True:
        train_start = cursor
        train_end = train_start + train_years * one_year
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + test_years * one_year
        if test_end > end:
            break
        segments.append((train_start, train_end, test_start, test_end))
        cursor = cursor + step_years * one_year
    return segments
