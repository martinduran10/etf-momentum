"""Performance and risk metrics for backtested return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(returns: pd.Series) -> float:
    """CAGR computed geometrically from daily returns."""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    cumulative = (1.0 + returns).prod()
    years = len(returns) / TRADING_DAYS_PER_YEAR
    return cumulative ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Sharpe ratio, daily returns annualized.

    Args:
        returns: Strategy daily returns.
        rf_annual: Annual risk-free rate (e.g. 0.04 for 4%).
    """
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = returns - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Sortino ratio — Sharpe analogue using downside deviation only."""
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_vol = np.sqrt((downside ** 2).mean())
    if downside_vol == 0:
        return float("nan")
    return (excess.mean() / downside_vol) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    """Max peak-to-trough drawdown of cumulative equity, as a negative fraction."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR / |max drawdown|."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return annualized_return(returns) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with strictly positive return."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def summary(returns: pd.Series, name: str = "strategy") -> pd.DataFrame:
    """One-row summary table of the standard metrics."""
    return pd.DataFrame(
        {
            "CAGR": annualized_return(returns),
            "Vol (ann.)": annualized_vol(returns),
            "Sharpe": sharpe_ratio(returns),
            "Sortino": sortino_ratio(returns),
            "Max DD": max_drawdown(returns),
            "Calmar": calmar_ratio(returns),
            "Hit rate": hit_rate(returns),
        },
        index=[name],
    )
