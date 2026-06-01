"""Overbought-market cash filter (Phase 2 overlay).

A precomputed binary signal flags days as overbought (``market_ok == 0``).
When a day ``T`` is flagged, the strategy moves to cash on the four trading
days ``T+3, T+4, T+5, T+6`` (counted in the strategy's trading-day calendar).
Overlapping windows merge naturally into continuous cash periods.

The overlay is applied to Phase 1's combined daily return series; the engine
itself is untouched. The signal is exogenous and precomputed — this module
loads it and translates it into a per-trading-day cash mask only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import DATA_DIR

#: Trading-day offsets forming the T+3..T+6 cash window.
CASH_LAG_RANGE = range(3, 7)


def load_overbought_signal(path: Path | str | None = None) -> pd.Series:
    """Load the precomputed overbought-market binary signal.

    Parameters
    ----------
    path : Path or str, optional
        Path to ``overbought_signal.csv``. Defaults to
        ``data/overbought_signal.csv``.

    Returns
    -------
    pandas.Series
        ``market_ok`` (``0``/``1``) indexed by date, sorted ascending.
    """
    if path is None:
        path = DATA_DIR / "overbought_signal.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()["market_ok"].astype(int)


def build_cash_mask(
    signal: pd.Series, trading_days: pd.DatetimeIndex
) -> pd.Series:
    """Build the daily cash mask in the strategy's trading-day calendar.

    A trading day ``t`` is a cash day when *any* of the signals
    ``signal[t-3], signal[t-4], signal[t-5], signal[t-6]`` equals 0 (offsets
    counted in ``trading_days`` space). Overlapping windows merge naturally.

    Parameters
    ----------
    signal : pandas.Series
        ``market_ok`` (``0``/``1``) indexed by date.
    trading_days : pandas.DatetimeIndex
        The strategy's trading-day calendar.

    Returns
    -------
    pandas.Series of bool
        ``True`` where the day is cash, indexed by ``trading_days``.

    Notes
    -----
    Edge handling:

    * Strategy trading days with no matching date in the filter CSV are
      treated as ``market_ok == 1`` (OK).
    * Lookback offsets that fall before the start of the filter data are
      treated as ``market_ok == 1`` (OK).

    No look-ahead: ``mask[t]`` depends only on ``signal`` values at trading
    days ``t-3`` and earlier — the 3-day minimum lag guarantees it.
    """
    aligned = signal.reindex(trading_days, fill_value=1).astype(int)
    cash = pd.Series(False, index=trading_days)
    for k in CASH_LAG_RANGE:
        lagged = aligned.shift(k).fillna(1).astype(int)
        cash |= lagged.eq(0)
    return cash


def apply_filter(
    phase1_returns: pd.Series, cash_mask: pd.Series
) -> pd.Series:
    """Force Phase 1 returns to zero on cash days.

    Parameters
    ----------
    phase1_returns : pandas.Series
        Combined daily returns from Phase 1.
    cash_mask : pandas.Series of bool
        ``True`` where the day should be in cash. Aligned to
        ``phase1_returns`` by reindex (missing days default to non-cash).

    Returns
    -------
    pandas.Series
        Filtered returns: ``0.0`` on cash days, original Phase 1 return
        otherwise. Same index as ``phase1_returns``.
    """
    mask = cash_mask.reindex(phase1_returns.index, fill_value=False)
    return phase1_returns.where(~mask, 0.0)
