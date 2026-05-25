"""Parameter sensitivity analysis for cross-sectional momentum.

Provides utilities to sweep over signal parameters (lookback, skip) and
portfolio parameters (n_long, costs) and produce summary tables of risk/return
metrics for each combination. Used to assess robustness of any chosen
parameter set and surface the parameter regions where the strategy works.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

from .backtest import run_backtest, RebalanceFreq
from .metrics import (
    annualized_return,
    annualized_vol,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
)
from .signals import momentum_total_return, momentum_risk_adjusted


def sweep_momentum_params(
    panel: pd.DataFrame,
    lookbacks: Sequence[int] = (3, 6, 9, 12),
    skips: Sequence[int] = (0, 1),
    n_longs: Sequence[int] = (5, 10, 15),
    rebalance: RebalanceFreq = "ME",
    cost_bps_per_side: float = 5.0,
    risk_adjusted: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Run the strategy across every combination of (lookback, skip, n_long).

    For each combination, computes CAGR, annualized vol, Sharpe, max drawdown
    and Calmar over the chosen window. The output is a long-format DataFrame
    ready for pivot tables or heatmaps.

    Args:
        panel: Long-format ETF panel.
        lookbacks: Lookback windows in months to sweep.
        skips: Skip windows in months to sweep.
        n_longs: Top-N portfolio sizes to sweep.
        rebalance: Rebalance frequency.
        cost_bps_per_side: Per-side transaction cost in bps.
        risk_adjusted: If True, use vol-scaled momentum signal.
        start: Optional start date for the backtest window.
        end: Optional end date.

    Returns:
        DataFrame with one row per (lookback, skip, n_long) combination
        and columns for the headline metrics.
    """
    signal_fn = momentum_risk_adjusted if risk_adjusted else momentum_total_return
    rows = []

    for lookback, skip in product(lookbacks, skips):
        if skip >= lookback:
            continue
        panel_with_sig = panel.copy()
        panel_with_sig["sig"] = signal_fn(panel_with_sig, lookback, skip)

        for n_long in n_longs:
            try:
                result = run_backtest(
                    panel_with_sig,
                    signal_col="sig",
                    n_long=n_long,
                    rebalance=rebalance,
                    cost_bps_per_side=cost_bps_per_side,
                    start=start,
                    end=end,
                )
            except Exception as e:  # noqa: BLE001
                rows.append({
                    "lookback": lookback, "skip": skip, "n_long": n_long,
                    "error": str(e),
                })
                continue

            r = result.daily_returns
            rows.append({
                "lookback": lookback,
                "skip": skip,
                "n_long": n_long,
                "cagr": annualized_return(r),
                "vol": annualized_vol(r),
                "sharpe": sharpe_ratio(r),
                "max_dd": max_drawdown(r),
                "calmar": calmar_ratio(r),
                "avg_turnover_pct": result.turnover.mean() * 100,
                "n_rebalances": len(result.rebalance_dates),
            })

    return pd.DataFrame(rows)


def pivot_for_heatmap(
    sweep: pd.DataFrame,
    metric: str = "sharpe",
    index: str = "lookback",
    columns: str = "n_long",
    skip: int | None = None,
) -> pd.DataFrame:
    """Reshape a sweep result into a 2D matrix for plotting heatmaps."""
    df = sweep.dropna(subset=[metric]) if metric in sweep.columns else sweep
    if skip is not None and "skip" in df.columns:
        df = df[df["skip"] == skip]
    return df.pivot_table(index=index, columns=columns, values=metric)
