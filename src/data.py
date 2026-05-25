"""Data loading and panel manipulation utilities.

The cleaned ETF panel lives in ``data/etf_panel.parquet`` as a long-format table
with one row per (date, ticker). Helper functions here load it, reshape it, and
filter it for downstream use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# Resolve the repo root so paths work from anywhere (notebooks, tests, scripts).
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def load_panel(path: Path | str | None = None) -> pd.DataFrame:
    """Load the long-format ETF panel.

    Args:
        path: Optional override for the panel file. Defaults to
            ``data/etf_panel.parquet``.

    Returns:
        DataFrame with columns
        ``[date, ticker, category, close, return, log_return, vol_10d_ann,
        signal_ret_over_vol, sheet_name]`` sorted by (ticker, date).
    """
    path = Path(path) if path else DATA_DIR / "etf_panel.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_universe(path: Path | str | None = None) -> pd.DataFrame:
    """Load the universe map (ticker → category → coverage)."""
    path = Path(path) if path else DATA_DIR / "universe.csv"
    return pd.read_csv(path, parse_dates=["start_date", "end_date"])


def load_wide_closes(path: Path | str | None = None) -> pd.DataFrame:
    """Load the wide-format price matrix (dates × tickers)."""
    path = Path(path) if path else DATA_DIR / "closes_wide.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.sort_index()


def to_wide(panel: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """Pivot a long-format panel into a wide (dates × tickers) frame.

    Args:
        panel: Long-format panel with at least ``date``, ``ticker`` and ``column``.
        column: Which column to pivot. Defaults to ``"close"``.
    """
    return panel.pivot_table(index="date", columns="ticker", values=column).sort_index()


def filter_tickers(panel: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Subset the panel to a specified list of tickers."""
    return panel[panel["ticker"].isin(set(tickers))].reset_index(drop=True)


def filter_dates(
    panel: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Subset the panel to a date range (inclusive)."""
    out = panel
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)
