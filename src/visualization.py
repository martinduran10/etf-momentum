"""Publication-quality plotting helpers.

All functions return a ``matplotlib.figure.Figure`` so callers can decide
whether to display, save or further customize. The style is deliberately
restrained — readable in print and on dark or light backgrounds, with
enough labeling that figures are self-contained for a report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Consistent color palette across the project
COLORS = {
    "strategy":  "#1f4e79",   # deep blue
    "improved":  "#2e8b57",   # forest green
    "benchmark": "#888888",   # neutral grey
    "spy":       "#c0392b",   # muted red
    "accent":    "#d4a017",   # gold
    "negative":  "#a52a2a",
}


def _apply_style() -> None:
    """Apply a clean, consistent matplotlib style."""
    plt.rcParams.update({
        "figure.figsize": (12, 5),
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "font.family": "sans-serif",
    })


def plot_equity_curves(
    curves: dict[str, pd.Series],
    title: str = "Equity curves",
    color_map: dict[str, str] | None = None,
    log_y: bool = False,
) -> plt.Figure:
    """Plot one or more equity curves on a single axis."""
    _apply_style()
    fig, ax = plt.subplots()
    color_map = color_map or {}
    for label, series in curves.items():
        color = color_map.get(label, None)
        ax.plot(series.index, series.values, label=label, color=color, linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel("cumulative return (start = 1.0)")
    if log_y:
        ax.set_yscale("log")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_drawdowns(
    returns_dict: dict[str, pd.Series],
    title: str = "Drawdowns",
    color_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot drawdown series for multiple strategies."""
    _apply_style()
    fig, ax = plt.subplots()
    color_map = color_map or {}
    for label, r in returns_dict.items():
        eq = (1 + r.fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        color = color_map.get(label, None)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color)
        ax.plot(dd.index, dd.values, label=label, color=color, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_sharpe_heatmap(
    matrix: pd.DataFrame,
    title: str = "Sharpe ratio by parameter combination",
    cmap: str = "RdYlGn",
    annot_fmt: str = ".2f",
) -> plt.Figure:
    """Heatmap of a metric (typically Sharpe) across two parameters."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = float(np.nanmax(np.abs(matrix.values)))
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(matrix.columns.name or "")
    ax.set_ylabel(matrix.index.name or "")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, format(v, annot_fmt), ha="center", va="center",
                        color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    daily_returns: pd.Series,
    title: str = "Monthly returns",
) -> plt.Figure:
    """Heatmap of monthly returns: rows = years, cols = months."""
    _apply_style()
    monthly = (1 + daily_returns.fillna(0)).resample("ME").prod() - 1
    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    }).pivot(index="year", columns="month", values="ret")
    months_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [months_lbl[c-1] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(pivot))))
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 0.05
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series],
    window_days: int = 252,
    title: str | None = None,
    color_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Rolling Sharpe ratio for multiple strategies."""
    _apply_style()
    title = title or f"Rolling {window_days}-day Sharpe"
    fig, ax = plt.subplots()
    color_map = color_map or {}
    for label, r in returns_dict.items():
        ann_factor = np.sqrt(252)
        rolling = r.rolling(window_days).mean() / r.rolling(window_days).std() * ann_factor
        ax.plot(rolling.index, rolling.values, label=label,
                color=color_map.get(label), linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Sharpe")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
