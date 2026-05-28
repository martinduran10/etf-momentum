"""Performance metrics for a non-compounding (arithmetic) return series.

Because the Stack Portfolio resets capital to $100 at every rebalance and
never compounds, the cumulative equity curve is the *running sum* of daily
returns, and the total return is the *sum* of daily returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def total_return(daily_returns: pd.Series) -> float:
    """Arithmetic total return: the sum of daily returns.

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily portfolio returns.

    Returns
    -------
    float
        ``sum(daily_returns)``.
    """
    return float(daily_returns.sum())


def annualized_return(daily_returns: pd.Series) -> float:
    """Annualized return: ``mean(daily_returns) * 252``.

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily portfolio returns.

    Returns
    -------
    float
        The annualized arithmetic return.
    """
    return float(daily_returns.mean() * TRADING_DAYS)


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized volatility: ``std(daily_returns, ddof=1) * sqrt(252)``.

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily portfolio returns.

    Returns
    -------
    float
        The annualized sample volatility.
    """
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series) -> float:
    """Sharpe ratio with a zero risk-free rate.

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily portfolio returns.

    Returns
    -------
    float
        ``annualized_return / annualized_volatility``.
    """
    vol = annualized_volatility(daily_returns)
    if vol == 0:
        return float("nan")
    return annualized_return(daily_returns) / vol


def equity_curve(daily_returns: pd.Series) -> pd.Series:
    """Cumulative arithmetic equity curve (running sum of daily returns).

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily portfolio returns.

    Returns
    -------
    pandas.Series
        The cumulative-sum curve, in return units (0.0 at inception).
    """
    return daily_returns.cumsum()


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Drawdown of the cumulative-arithmetic-return curve.

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily portfolio returns.

    Returns
    -------
    pandas.Series
        ``curve - running_max(curve)``; values are <= 0.
    """
    curve = equity_curve(daily_returns)
    running_max = curve.cummax()
    return curve - running_max


def max_drawdown(daily_returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the cumulative-return curve.

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily portfolio returns.

    Returns
    -------
    float
        The maximum drawdown as a negative number (e.g. ``-0.2849``).
    """
    return float(drawdown_series(daily_returns).min())


def summary(daily_returns: pd.Series) -> dict[str, float]:
    """Compute all headline metrics for a daily return series.

    Parameters
    ----------
    daily_returns : pandas.Series
        Daily portfolio returns.

    Returns
    -------
    dict of str to float
        Keys: ``total_return``, ``annualized_return``,
        ``annualized_volatility``, ``sharpe_ratio``, ``max_drawdown``.
    """
    return {
        "total_return": total_return(daily_returns),
        "annualized_return": annualized_return(daily_returns),
        "annualized_volatility": annualized_volatility(daily_returns),
        "sharpe_ratio": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(daily_returns),
    }
