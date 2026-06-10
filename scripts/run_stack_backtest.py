"""End-to-end Stack Portfolio backtest runner.

Loads the close-price data, runs the Stack Portfolio, writes the headline and
per-sub-strategy metrics to ``reports/tables/`` and the equity-curve and
drawdown figures to ``reports/figures/``.

Usage
-----
    python scripts/run_stack_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / scripted runs
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmarks import (  # noqa: E402
    benchmark_summary,
    compounded_equity_curve,
    spy_buy_and_hold_returns,
)
from src.data import load_closes  # noqa: E402
from src.individual_overbought import run_individual_overbought  # noqa: E402
from src.metrics import (  # noqa: E402
    drawdown_series,
    equity_curve,
    summary,
)
from src.overbought_filter import (  # noqa: E402
    apply_filter,
    build_cash_mask,
    load_overbought_signal,
)
from src.stack_backtest import (  # noqa: E402
    HEADLINE_START,
    run_stack_portfolio,
)

FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"


def _save_equity_curve(returns: pd.Series, path: Path) -> None:
    curve = equity_curve(returns) * 100.0
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(curve.index, curve.values, color="#1f4e79", lw=1.4)
    ax.set_title("Stack Portfolio — Cumulative Arithmetic Return")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Date")
    ax.axhline(0, color="grey", lw=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_drawdown(returns: pd.Series, path: Path) -> None:
    dd = drawdown_series(returns) * 100.0
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd.index, dd.values, 0.0, color="#b22222", alpha=0.5)
    ax.plot(dd.index, dd.values, color="#b22222", lw=0.8)
    ax.set_title("Stack Portfolio — Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_filtered_vs_unfiltered(
    unfiltered: pd.Series, filtered: pd.Series, path: Path
) -> None:
    unfiltered_curve = equity_curve(unfiltered) * 100.0
    filtered_curve = equity_curve(filtered) * 100.0
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        unfiltered_curve.index,
        unfiltered_curve.values,
        color="#1f4e79",
        lw=1.4,
        label="Phase 1 (unfiltered)",
    )
    ax.plot(
        filtered_curve.index,
        filtered_curve.values,
        color="#2a8f3a",
        lw=1.4,
        label="Phase 2 (overbought filter)",
    )
    ax.set_title("Stack Portfolio — Overbought Filter vs Unfiltered")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Date")
    ax.axhline(0, color="grey", lw=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_phase2b_comparison(
    phase1: pd.Series,
    phase12: pd.Series,
    phase2b: pd.Series,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for returns, color, label in (
        (phase1, "#1f4e79", "Phase 1"),
        (phase12, "#2a8f3a", "Phase 1+2 (market filter)"),
        (phase2b, "#9b2f8f", "Phase 2b (individual screen)"),
    ):
        curve = equity_curve(returns) * 100.0
        ax.plot(curve.index, curve.values, color=color, lw=1.4, label=label)
    ax.set_title("Stack Portfolio — Phase 2b vs Phase 1 / Phase 1+2")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Date")
    ax.axhline(0, color="grey", lw=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_stack_vs_spy(
    stack_returns: pd.Series, spy_returns: pd.Series, path: Path
) -> None:
    stack_curve = equity_curve(stack_returns) * 100.0  # arithmetic (no compounding)
    spy_curve = compounded_equity_curve(spy_returns) * 100.0  # buy-and-hold
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        stack_curve.index,
        stack_curve.values,
        color="#1f4e79",
        lw=1.4,
        label="Stack Portfolio (arithmetic)",
    )
    ax.plot(
        spy_curve.index,
        spy_curve.values,
        color="#c0700f",
        lw=1.4,
        label="SPY buy-and-hold (compounded)",
    )
    ax.set_title("Stack Portfolio vs SPY")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Date")
    ax.axhline(0, color="grey", lw=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    closes = load_closes()
    result = run_stack_portfolio(closes)
    headline = result["headline_returns"]

    # Headline metrics table.
    metrics = summary(headline)
    metrics_df = pd.DataFrame(
        {"metric": list(metrics), "value": list(metrics.values())}
    )
    metrics_df.to_csv(TABLES_DIR / "metrics_summary.csv", index=False)

    # Per-sub-strategy metrics (measured from each sub's own start onward).
    sub_rows = []
    for name, res in result["sub_results"].items():
        sub_returns = res.returns.iloc[res.start_idx :]
        s = summary(sub_returns)
        sub_rows.append({"sub_strategy": name, **s})
    pd.DataFrame(sub_rows).to_csv(
        TABLES_DIR / "sub_strategy_metrics.csv", index=False
    )

    # SPY buy-and-hold benchmark comparison.
    spy_returns = spy_buy_and_hold_returns(closes)
    spy_metrics = benchmark_summary(spy_returns)
    comparison = pd.DataFrame(
        {"Stack Portfolio": metrics, "SPY": spy_metrics}
    )
    comparison.index.name = "metric"
    comparison.to_csv(TABLES_DIR / "spy_comparison.csv")

    # Phase 2: overbought-market cash filter overlay.
    signal = load_overbought_signal()
    cash_mask = build_cash_mask(signal, closes.index)
    filtered_headline = apply_filter(headline, cash_mask)
    filtered_metrics = summary(filtered_headline)
    cash_share = float(
        cash_mask.reindex(headline.index, fill_value=False).mean()
    )

    phase2_comparison = pd.DataFrame(
        {
            "Unfiltered (Phase 1)": metrics,
            "Filtered (Phase 2)": filtered_metrics,
        }
    )
    phase2_comparison.loc["pct_days_in_cash"] = [float("nan"), cash_share]
    phase2_comparison.index.name = "metric"
    phase2_comparison.to_csv(TABLES_DIR / "phase2_comparison.csv")

    # Phase 2b: individual-ETF overbought filter (daily-rebalanced overlay).
    # Main series: daily rank + individual screen, then the Phase 2 market mask.
    phase2b = run_individual_overbought(closes, screen=True)
    phase2b_headline = phase2b["headline_returns"]
    phase2b_filtered = apply_filter(phase2b_headline, cash_mask)
    phase2b_metrics = summary(phase2b_filtered)

    # Diagnostic: daily-rebalanced, individual screen OFF, no market mask —
    # isolates the daily-rebalance change from the filter's effect.
    diag = run_individual_overbought(closes, screen=False)
    diag_headline = diag["headline_returns"]
    diag_metrics = summary(diag_headline)

    phase2b_comparison = pd.DataFrame(
        {
            "Phase 1": metrics,
            "Phase 1+2": filtered_metrics,
            "Phase 2b": phase2b_metrics,
            "Daily-rebal (screen off)": diag_metrics,
        }
    )
    phase2b_comparison.index.name = "metric"
    phase2b_comparison.to_csv(TABLES_DIR / "phase2b_comparison.csv")

    # Figures.
    _save_equity_curve(headline, FIGURES_DIR / "equity_curve.png")
    _save_drawdown(headline, FIGURES_DIR / "drawdown.png")
    _save_stack_vs_spy(headline, spy_returns, FIGURES_DIR / "stack_vs_spy.png")
    _save_filtered_vs_unfiltered(
        headline, filtered_headline, FIGURES_DIR / "stack_filtered_vs_unfiltered.png"
    )
    _save_phase2b_comparison(
        headline,
        filtered_headline,
        phase2b_filtered,
        FIGURES_DIR / "phase2b_comparison.png",
    )

    # Console summary (script entry point only; library code stays silent).
    span = f"{headline.index[0].date()} -> {headline.index[-1].date()}"
    lines = [
        f"Stack Portfolio headline ({HEADLINE_START} onward, {span}):",
        f"  Total return       : {metrics['total_return']:.2%}",
        f"  Annualized return  : {metrics['annualized_return']:.2%}",
        f"  Annualized vol     : {metrics['annualized_volatility']:.2%}",
        f"  Sharpe ratio       : {metrics['sharpe_ratio']:.2f}",
        f"  Max drawdown       : {metrics['max_drawdown']:.2%}",
        "SPY buy-and-hold (compounded) over the same window:",
        f"  Total return       : {spy_metrics['total_return']:.2%}",
        f"  Annualized return  : {spy_metrics['annualized_return']:.2%}",
        f"  Annualized vol     : {spy_metrics['annualized_volatility']:.2%}",
        f"  Sharpe ratio       : {spy_metrics['sharpe_ratio']:.2f}",
        f"  Max drawdown       : {spy_metrics['max_drawdown']:.2%}",
        "Phase 2 (overbought filter, T+3..T+6 cash) over the same window:",
        f"  Total return       : {filtered_metrics['total_return']:.2%}",
        f"  Annualized return  : {filtered_metrics['annualized_return']:.2%}",
        f"  Annualized vol     : {filtered_metrics['annualized_volatility']:.2%}",
        f"  Sharpe ratio       : {filtered_metrics['sharpe_ratio']:.2f}",
        f"  Max drawdown       : {filtered_metrics['max_drawdown']:.2%}",
        f"  Days in cash       : {cash_share:.2%}",
        "Phase 2b (individual screen + market mask) over the same window:",
        f"  Total return       : {phase2b_metrics['total_return']:.2%}",
        f"  Annualized return  : {phase2b_metrics['annualized_return']:.2%}",
        f"  Annualized vol     : {phase2b_metrics['annualized_volatility']:.2%}",
        f"  Sharpe ratio       : {phase2b_metrics['sharpe_ratio']:.2f}",
        f"  Max drawdown       : {phase2b_metrics['max_drawdown']:.2%}",
        "Diagnostic — daily-rebalanced, screen off, no market mask:",
        f"  Total return       : {diag_metrics['total_return']:.2%}",
        f"  Annualized return  : {diag_metrics['annualized_return']:.2%}",
        f"  Annualized vol     : {diag_metrics['annualized_volatility']:.2%}",
        f"  Sharpe ratio       : {diag_metrics['sharpe_ratio']:.2f}",
        f"  Max drawdown       : {diag_metrics['max_drawdown']:.2%}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
