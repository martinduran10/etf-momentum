"""Momentum signal computation.

Implements several flavors of cross-sectional momentum:

* ``momentum_total_return`` — classic Jegadeesh-Titman style trailing return,
  optionally skipping the most recent month to avoid short-term reversal.
* ``momentum_risk_adjusted`` — trailing return divided by realized volatility
  over the same window.
* ``daily_signal_ret_over_vol`` — reproduces the original Excel signal,
  one-day return divided by 10-day annualized volatility.

All functions operate on the long-format panel (one row per date / ticker)
and return a new column added to the panel. They are deliberately written
to avoid any look-ahead bias: the signal at time ``t`` only uses data from
``t`` and earlier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252


def momentum_total_return(
    panel: pd.DataFrame,
    lookback_months: int = 6,
    skip_months: int = 1,
) -> pd.Series:
    """Trailing total return over ``lookback_months``, skipping the most
    recent ``skip_months``.

    Standard academic momentum: e.g. ``lookback_months=12, skip_months=1``
    gives the canonical "12-1" momentum used in Jegadeesh-Titman (1993)
    and downstream literature. The skip avoids the well-documented one-month
    short-term reversal effect.

    Args:
        panel: Long-format panel with ``date``, ``ticker``, ``close``.
        lookback_months: Total lookback length in months.
        skip_months: Most recent months to exclude from the signal.

    Returns:
        Series aligned with the input panel index containing the signal.
    """
    if skip_months >= lookback_months:
        raise ValueError("skip_months must be strictly less than lookback_months")

    panel = panel.sort_values(["ticker", "date"])
    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    skip_days = skip_months * TRADING_DAYS_PER_MONTH

    # Price t-skip / price t-lookback - 1
    grp = panel.groupby("ticker")["close"]
    price_lagged = grp.shift(skip_days)
    price_base = grp.shift(lookback_days)
    return (price_lagged / price_base - 1.0).rename(
        f"mom_{lookback_months}m_skip_{skip_months}m"
    )


def momentum_risk_adjusted(
    panel: pd.DataFrame,
    lookback_months: int = 6,
    skip_months: int = 1,
) -> pd.Series:
    """Risk-adjusted momentum: trailing return divided by trailing volatility.

    This normalizes the raw momentum signal by realized risk over the same
    window, which has the effect of dampening signals from high-vol names
    and amplifying signals from steady trends.
    """
    if skip_months >= lookback_months:
        raise ValueError("skip_months must be strictly less than lookback_months")

    panel = panel.sort_values(["ticker", "date"])
    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    skip_days = skip_months * TRADING_DAYS_PER_MONTH

    raw = momentum_total_return(panel, lookback_months, skip_months)

    # Realized vol over the same window (excluding the skip period)
    def _trailing_vol(s: pd.Series) -> pd.Series:
        shifted = s.shift(skip_days)
        return shifted.rolling(lookback_days - skip_days).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

    trailing_vol = panel.groupby("ticker")["log_return"].transform(_trailing_vol)
    return (raw / trailing_vol).rename(
        f"mom_{lookback_months}m_skip_{skip_months}m_riskadj"
    )


def daily_signal_ret_over_vol(panel: pd.DataFrame) -> pd.Series:
    """Reproduces the original Excel signal: daily return / 10-day annualized vol.

    Note: this is a noisy daily-frequency signal and not a true momentum
    signal in the academic sense — kept here for parity with the source
    Excel model.
    """
    return (panel["return"] / panel["vol_10d_ann"]).rename("signal_ret_over_vol")


def cross_sectional_rank(
    panel: pd.DataFrame,
    signal_col: str,
    ascending: bool = False,
) -> pd.Series:
    """Rank tickers cross-sectionally on each date by ``signal_col``.

    Args:
        panel: Long panel including the signal column.
        signal_col: Name of the column to rank.
        ascending: If False (default), higher signal → rank 1 (the winner).

    Returns:
        Integer rank series aligned with the input panel.
    """
    return (
        panel.groupby("date")[signal_col]
        .rank(method="dense", ascending=ascending, na_option="bottom")
        .astype("Int64")
        .rename(f"{signal_col}_rank")
    )
