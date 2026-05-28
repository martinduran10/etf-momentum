"""Signal construction for the Stack Portfolio.

Two signals are derived from raw closes:

* ``risk_adj_return`` — the daily log return divided by its 10-day annualized
  volatility.
* ``slow_signal`` — a rolling mean of ``risk_adj_return`` over a sub-strategy's
  lookback window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import compute_log_returns, compute_vol_10


def compute_risk_adj_return(closes: pd.DataFrame) -> pd.DataFrame:
    """Compute the daily risk-adjusted return signal per ETF.

    ``risk_adj_return_t = log_ret_t / vol_10_t`` where ``vol_10`` is the
    10-day annualized volatility of log returns. Where ``vol_10`` is zero or
    undefined the result is NaN.

    Parameters
    ----------
    closes : pandas.DataFrame
        Close prices indexed by date, one column per ticker.

    Returns
    -------
    pandas.DataFrame
        The ``risk_adj_return`` signal, same shape as ``closes``.
    """
    log_ret = compute_log_returns(closes)
    vol_10 = compute_vol_10(log_ret)
    # Avoid division-by-zero warnings; zeros become inf then are masked to NaN.
    vol_10 = vol_10.replace(0.0, np.nan)
    return log_ret / vol_10


def compute_slow_signal(
    risk_adj_return: pd.DataFrame, lookback: int
) -> pd.DataFrame:
    """Compute the slow signal: a rolling mean of ``risk_adj_return``.

    Parameters
    ----------
    risk_adj_return : pandas.DataFrame
        The daily risk-adjusted return signal per ETF.
    lookback : int
        Rolling window length in trading days.

    Returns
    -------
    pandas.DataFrame
        ``rolling_mean(risk_adj_return, lookback)`` per ETF. Rows before the
        window fills are NaN.
    """
    return risk_adj_return.rolling(window=lookback).mean()
