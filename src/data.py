"""Data loading and base return/volatility computation.

All series are recomputed from the raw wide-format close prices in
``data/closes_wide.csv``. Precomputed columns in ``etf_panel.parquet`` are
deliberately ignored, per the Phase 1 specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Repository root, resolved relative to this file so callers can run from
# anywhere.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

#: Annualization factor for daily data.
TRADING_DAYS = 252

#: Rolling window (trading days) for the realized-volatility estimate.
VOL_WINDOW = 10


def load_closes(path: Path | str | None = None) -> pd.DataFrame:
    """Load the wide-format daily close-price matrix.

    Parameters
    ----------
    path : Path or str, optional
        Path to the wide-format CSV. Defaults to ``data/closes_wide.csv``.

    Returns
    -------
    pandas.DataFrame
        Close prices indexed by trading date (``DatetimeIndex``, sorted
        ascending) with one column per ETF ticker.
    """
    if path is None:
        path = DATA_DIR / "closes_wide.csv"
    closes = pd.read_csv(path, parse_dates=["date"])
    closes = closes.set_index("date").sort_index()
    closes.columns = [c.lower() for c in closes.columns]
    return closes


def load_universe(path: Path | str | None = None) -> pd.DataFrame:
    """Load the 43-ETF universe map.

    Parameters
    ----------
    path : Path or str, optional
        Path to ``universe.csv``. Defaults to ``data/universe.csv``.

    Returns
    -------
    pandas.DataFrame
        The universe table as stored on disk.
    """
    if path is None:
        path = DATA_DIR / "universe.csv"
    return pd.read_csv(path)


def compute_log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns per ETF.

    Parameters
    ----------
    closes : pandas.DataFrame
        Close prices indexed by date, one column per ticker.

    Returns
    -------
    pandas.DataFrame
        ``ln(close_t / close_{t-1})`` per ETF. The first row is NaN.
    """
    return np.log(closes / closes.shift(1))


def compute_simple_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple (close-to-close) returns per ETF.

    These are the returns earned by an invested slot, used for the
    dollar-based, non-compounding P&L of the backtest.

    Parameters
    ----------
    closes : pandas.DataFrame
        Close prices indexed by date, one column per ticker.

    Returns
    -------
    pandas.DataFrame
        ``close_t / close_{t-1} - 1`` per ETF. The first row is NaN.
    """
    return closes.pct_change(fill_method=None)


def compute_vol_10(
    log_returns: pd.DataFrame, window: int = VOL_WINDOW
) -> pd.DataFrame:
    """Annualized rolling volatility of log returns.

    Parameters
    ----------
    log_returns : pandas.DataFrame
        Daily log returns per ETF.
    window : int, optional
        Rolling window length in trading days (default 10).

    Returns
    -------
    pandas.DataFrame
        ``rolling_std(log_ret, window) * sqrt(252)`` per ETF, using the
        sample standard deviation (``ddof=1``) to match the Excel ``STDEV``
        convention. Values before the window fills are NaN.
    """
    return log_returns.rolling(window=window).std(ddof=1) * np.sqrt(TRADING_DAYS)
