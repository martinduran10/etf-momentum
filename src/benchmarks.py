"""Buy-and-hold benchmark comparison (SPY).

The Stack Portfolio is a non-compounding (arithmetic) strategy, so its metrics
come from :mod:`src.metrics`. A buy-and-hold position in SPY, by contrast, is
naturally compounded: profits stay invested. This module computes SPY's
compounded return series and compounding-consistent metrics so the two can be
compared side by side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS
from .metrics import annualized_volatility
from .stack_backtest import HEADLINE_START


def spy_buy_and_hold_returns(
    closes: pd.DataFrame, start: str = HEADLINE_START
) -> pd.Series:
    """Daily simple returns of a SPY buy-and-hold position.

    The position is entered at the ``start`` date's close, so the return on the
    start date itself is zero and the series is aligned to the same trading
    calendar (and length) as the headline strategy returns.

    Parameters
    ----------
    closes : pandas.DataFrame
        Close prices indexed by date, including a ``spy`` column.
    start : str, optional
        Inclusive start date (default: the headline start, ``2015-12-07``).

    Returns
    -------
    pandas.Series
        Daily simple returns of SPY from ``start`` onward; the first value is 0.
    """
    spy = closes["spy"].loc[start:]
    return spy.pct_change(fill_method=None).fillna(0.0)


def compounded_equity_curve(daily_returns: pd.Series) -> pd.Series:
    """Compounded cumulative-return curve (buy-and-hold growth of $1, minus 1).

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily simple returns.

    Returns
    -------
    pandas.Series
        ``(1 + r).cumprod() - 1``, in return units (0.0 at inception).
    """
    return (1.0 + daily_returns).cumprod() - 1.0


def compounded_max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum drawdown of the compounded wealth curve.

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily simple returns.

    Returns
    -------
    float
        The largest peak-to-trough decline as a negative number, measured as
        ``wealth / running_max(wealth) - 1``.
    """
    wealth = (1.0 + daily_returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def benchmark_summary(daily_returns: pd.Series) -> dict[str, float]:
    """Compounding-consistent headline metrics for a buy-and-hold series.

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily simple returns.

    Returns
    -------
    dict of str to float
        Keys: ``total_return`` (compounded), ``annualized_return`` (CAGR),
        ``annualized_volatility``, ``sharpe_ratio`` (CAGR / vol, rf=0),
        ``max_drawdown`` (from the compounded curve).
    """
    n = len(daily_returns)
    total = float((1.0 + daily_returns).prod() - 1.0)
    cagr = (1.0 + total) ** (TRADING_DAYS / n) - 1.0 if n else float("nan")
    vol = annualized_volatility(daily_returns)
    sharpe = cagr / vol if vol else float("nan")
    return {
        "total_return": total,
        "annualized_return": float(cagr),
        "annualized_volatility": vol,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": compounded_max_drawdown(daily_returns),
    }
