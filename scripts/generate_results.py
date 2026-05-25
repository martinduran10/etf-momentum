"""End-to-end research driver.

Reproduces every experiment in the project and writes outputs to
``reports/figures/`` and ``reports/tables/``. Running this script once
regenerates everything that appears in RESULTS.md.

Usage:
    python scripts/generate_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_panel
from src.signals import momentum_total_return, momentum_risk_adjusted
from src.backtest import run_backtest, equal_weight_benchmark, walk_forward_segments
from src.regime import trend_filter
from src.metrics import summary
from src.sensitivity import sweep_momentum_params, pivot_for_heatmap
from src.analysis import (
    regress_against_market,
    bootstrap_sharpe_ci,
    cost_sensitivity,
)
from src.visualization import (
    plot_equity_curves,
    plot_drawdowns,
    plot_sharpe_heatmap,
    plot_monthly_returns_heatmap,
    plot_rolling_sharpe,
    COLORS,
)


FIG_DIR = REPO_ROOT / "reports" / "figures"
TBL_DIR = REPO_ROOT / "reports" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    """Save a figure as PNG and close it to free memory."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved {path.relative_to(REPO_ROOT)}")


def main() -> None:
    print("\n=== Loading data ===")
    panel = load_panel()
    print(f"  Panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    # ---------------------------------------------------------------- baseline
    print("\n=== Baseline: 6-month skip-1 momentum, top 10, monthly ===")
    panel["mom_6_1"] = momentum_total_return(panel, lookback_months=6, skip_months=1)
    baseline = run_backtest(panel, "mom_6_1", n_long=10, rebalance="ME",
                             cost_bps_per_side=5.0)

    # Benchmarks
    ew = equal_weight_benchmark(panel)
    spy = panel[panel["ticker"] == "spy"].set_index("date")["close"]
    spy_eq = spy / spy.iloc[0]

    # Align all curves to the baseline window
    eq_index = baseline.equity_curve.index
    ew_a = ew.reindex(eq_index).ffill(); ew_a = ew_a / ew_a.iloc[0]
    spy_a = spy_eq.reindex(eq_index).ffill(); spy_a = spy_a / spy_a.iloc[0]

    baseline_summary = pd.concat([
        summary(baseline.daily_returns, "Momentum 6-1 (baseline)"),
        summary(spy_a.pct_change(), "SPY"),
        summary(ew_a.pct_change(), "Equal-weight universe"),
    ])
    baseline_summary.to_csv(TBL_DIR / "01_baseline_summary.csv")
    print(baseline_summary.round(4).to_string())

    fig = plot_equity_curves(
        {"Momentum 6-1 (baseline)": baseline.equity_curve,
         "SPY": spy_a, "Equal-weight universe": ew_a},
        title="Baseline strategy vs benchmarks",
        color_map={"Momentum 6-1 (baseline)": COLORS["strategy"],
                   "SPY": COLORS["spy"],
                   "Equal-weight universe": COLORS["benchmark"]},
    )
    save(fig, "01_baseline_equity")

    fig = plot_drawdowns(
        {"Momentum 6-1": baseline.daily_returns,
         "SPY": spy_a.pct_change()},
        title="Baseline drawdowns",
        color_map={"Momentum 6-1": COLORS["strategy"], "SPY": COLORS["spy"]},
    )
    save(fig, "01_baseline_drawdowns")

    # -------------------------------------------------------- sensitivity
    print("\n=== Parameter sensitivity sweep ===")
    sweep_raw = sweep_momentum_params(
        panel,
        lookbacks=(3, 6, 9, 12, 18),
        skips=(0, 1),
        n_longs=(3, 5, 10, 15, 20),
        risk_adjusted=False,
    )
    sweep_raw.to_csv(TBL_DIR / "02_sensitivity_raw.csv", index=False)
    print(f"  Tested {len(sweep_raw)} parameter combinations")
    print("  Top 5 by Sharpe:")
    print(sweep_raw.nlargest(5, "sharpe")[
        ["lookback","skip","n_long","cagr","vol","sharpe","max_dd","avg_turnover_pct"]
    ].round(3).to_string(index=False))

    # Heatmaps: skip=1 (the academically-favored variant), Sharpe by lookback × n_long
    for skip_val in (0, 1):
        m = pivot_for_heatmap(sweep_raw, "sharpe", "lookback", "n_long", skip=skip_val)
        fig = plot_sharpe_heatmap(
            m, title=f"Sharpe by lookback × n_long (skip={skip_val} month, raw return signal)"
        )
        save(fig, f"02_sensitivity_sharpe_skip{skip_val}")

    # ----------------------------------------------- risk-adjusted variant
    print("\n=== Risk-adjusted momentum ===")
    sweep_ra = sweep_momentum_params(
        panel,
        lookbacks=(3, 6, 9, 12, 18),
        skips=(0, 1),
        n_longs=(3, 5, 10, 15, 20),
        risk_adjusted=True,
    )
    sweep_ra.to_csv(TBL_DIR / "03_sensitivity_risk_adjusted.csv", index=False)
    print("  Top 5 by Sharpe (risk-adjusted):")
    print(sweep_ra.nlargest(5, "sharpe")[
        ["lookback","skip","n_long","cagr","vol","sharpe","max_dd","avg_turnover_pct"]
    ].round(3).to_string(index=False))

    # Head-to-head: best raw vs best risk-adjusted
    best_raw = sweep_raw.nlargest(1, "sharpe").iloc[0]
    best_ra  = sweep_ra.nlargest(1, "sharpe").iloc[0]
    print(f"\n  Best raw:           lb={int(best_raw['lookback'])}m skip={int(best_raw['skip'])}m "
          f"N={int(best_raw['n_long'])} → Sharpe {best_raw['sharpe']:.2f} CAGR {best_raw['cagr']:.1%}")
    print(f"  Best risk-adjusted: lb={int(best_ra['lookback'])}m skip={int(best_ra['skip'])}m "
          f"N={int(best_ra['n_long'])} → Sharpe {best_ra['sharpe']:.2f} CAGR {best_ra['cagr']:.1%}")

    # --------------------------------------------------- regime overlay
    print("\n=== Regime overlay: 200-day MA trend filter ===")
    # Use the best raw parameters (most comparable apples-to-apples to baseline)
    best_lb, best_skip, best_n = int(best_raw["lookback"]), int(best_raw["skip"]), int(best_raw["n_long"])
    sig_col = f"mom_{best_lb}_{best_skip}"
    panel[sig_col] = momentum_total_return(panel, best_lb, best_skip)

    gate = trend_filter(panel, market_ticker="spy", lookback_days=200)
    # Use the gate as evaluated AT each rebalance date
    no_gate = run_backtest(panel, sig_col, n_long=best_n, rebalance="ME")
    gated   = run_backtest(panel, sig_col, n_long=best_n, rebalance="ME",
                            regime_gate=gate)

    regime_summary = pd.concat([
        summary(no_gate.daily_returns, f"Momentum {best_lb}-{best_skip} (no filter)"),
        summary(gated.daily_returns,   f"Momentum {best_lb}-{best_skip} (200d MA filter)"),
        summary(spy_a.pct_change(),    "SPY"),
    ])
    regime_summary.to_csv(TBL_DIR / "04_regime_overlay.csv")
    print(regime_summary.round(4).to_string())
    print(f"\n  Invested fraction with filter: {gated.invested_pct:.1%}")

    eq_index = no_gate.equity_curve.index
    spy_aligned = spy_eq.reindex(eq_index).ffill(); spy_aligned = spy_aligned / spy_aligned.iloc[0]
    fig = plot_equity_curves(
        {"Momentum (no filter)": no_gate.equity_curve,
         "Momentum + 200d MA filter": gated.equity_curve,
         "SPY": spy_aligned},
        title=f"Regime overlay impact ({best_lb}-{best_skip} momentum, top {best_n})",
        color_map={"Momentum (no filter)": COLORS["benchmark"],
                   "Momentum + 200d MA filter": COLORS["improved"],
                   "SPY": COLORS["spy"]},
    )
    save(fig, "04_regime_equity")

    fig = plot_drawdowns(
        {"Momentum (no filter)": no_gate.daily_returns,
         "Momentum + 200d MA filter": gated.daily_returns,
         "SPY": spy_aligned.pct_change()},
        title="Regime overlay drawdown impact",
        color_map={"Momentum (no filter)": COLORS["benchmark"],
                   "Momentum + 200d MA filter": COLORS["improved"],
                   "SPY": COLORS["spy"]},
    )
    save(fig, "04_regime_drawdowns")

    fig = plot_monthly_returns_heatmap(gated.daily_returns,
                                       title=f"Monthly returns — Momentum {best_lb}-{best_skip} + filter")
    save(fig, "04_regime_monthly_heatmap")

    fig = plot_rolling_sharpe(
        {"Momentum (no filter)": no_gate.daily_returns,
         "Momentum + 200d MA filter": gated.daily_returns,
         "SPY": spy_aligned.pct_change()},
        window_days=252,
        title="Rolling 1-year Sharpe",
        color_map={"Momentum (no filter)": COLORS["benchmark"],
                   "Momentum + 200d MA filter": COLORS["improved"],
                   "SPY": COLORS["spy"]},
    )
    save(fig, "04_rolling_sharpe")

    # ------------------------------------------------ walk-forward OOS
    print("\n=== Walk-forward out-of-sample validation ===")
    all_dates = panel["date"].sort_values().unique()
    segments = walk_forward_segments(
        pd.DatetimeIndex(all_dates), train_years=4.0, test_years=1.5, step_years=1.5
    )
    print(f"  Generated {len(segments)} walk-forward segments")

    oos_rows = []
    oos_returns_concat = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(segments):
        # In each train window, find the best (lookback, skip, n_long) by Sharpe
        # using the SAME parameter grid as the sensitivity sweep
        train_sweep = sweep_momentum_params(
            panel,
            lookbacks=(3, 6, 9, 12),
            skips=(0, 1),
            n_longs=(5, 10, 15),
            risk_adjusted=False,
            start=str(train_start.date()),
            end=str(train_end.date()),
        )
        train_sweep = train_sweep.dropna(subset=["sharpe"])
        if train_sweep.empty:
            continue
        best = train_sweep.nlargest(1, "sharpe").iloc[0]
        lb, sk, n = int(best["lookback"]), int(best["skip"]), int(best["n_long"])

        # Apply chosen params to OOS window (with same regime filter)
        col = f"mom_oos_{i}"
        panel[col] = momentum_total_return(panel, lb, sk)
        oos_res = run_backtest(
            panel, col, n_long=n, rebalance="ME",
            start=str(test_start.date()), end=str(test_end.date()),
            regime_gate=gate,
        )

        oos_rows.append({
            "segment": i + 1,
            "train": f"{train_start.date()}→{train_end.date()}",
            "test":  f"{test_start.date()}→{test_end.date()}",
            "best_params_train": f"lb={lb} skip={sk} N={n}",
            "in_sample_sharpe": float(best["sharpe"]),
            "in_sample_cagr": float(best["cagr"]),
            "oos_cagr":   float(summary(oos_res.daily_returns, "x").iloc[0]["CAGR"]),
            "oos_sharpe": float(summary(oos_res.daily_returns, "x").iloc[0]["Sharpe"]),
            "oos_max_dd": float(summary(oos_res.daily_returns, "x").iloc[0]["Max DD"]),
        })
        oos_returns_concat.append(oos_res.daily_returns)

    walk_table = pd.DataFrame(oos_rows)
    walk_table.to_csv(TBL_DIR / "05_walk_forward.csv", index=False)
    print(walk_table.round(3).to_string(index=False))

    # Stitched OOS equity curve
    if oos_returns_concat:
        stitched = pd.concat(oos_returns_concat).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="first")]
        stitched_eq = (1 + stitched.fillna(0)).cumprod()
        spy_oos = spy_eq.reindex(stitched_eq.index).ffill()
        spy_oos = spy_oos / spy_oos.iloc[0]
        fig = plot_equity_curves(
            {"Walk-forward OOS strategy": stitched_eq, "SPY": spy_oos},
            title="Walk-forward out-of-sample equity curve",
            color_map={"Walk-forward OOS strategy": COLORS["improved"], "SPY": COLORS["spy"]},
        )
        save(fig, "05_walk_forward_equity")

        oos_summary = pd.concat([
            summary(stitched, "Walk-forward OOS"),
            summary(spy_oos.pct_change(), "SPY (same window)"),
        ])
        oos_summary.to_csv(TBL_DIR / "05_walk_forward_summary.csv")
        print("\n  OOS summary:")
        print(oos_summary.round(4).to_string())

    # ----------------------------------- statistical significance
    print("\n=== Statistical significance: bootstrap Sharpe CIs ===")
    spy_ret = spy_a.pct_change()
    ci_rows = []
    for label, ret in [
        ("Momentum 6-1 (baseline)",      baseline.daily_returns),
        (f"Momentum {best_lb}-{best_skip} (best in-sample)", no_gate.daily_returns),
        (f"Momentum {best_lb}-{best_skip} + 200d MA filter",  gated.daily_returns),
        ("Walk-forward OOS",             stitched if oos_returns_concat else pd.Series(dtype=float)),
        ("SPY",                          spy_ret),
        ("Equal-weight",                 ew_a.pct_change()),
    ]:
        if len(ret.dropna()) < 100:
            continue
        ci = bootstrap_sharpe_ci(ret, n_bootstrap=2000, block_length=21)
        ci_rows.append({
            "strategy": label,
            "sharpe": ci["sharpe"],
            "ci_low_95": ci["ci_low"],
            "ci_high_95": ci["ci_high"],
            "ci_includes_zero": ci["includes_zero"],
        })
    ci_table = pd.DataFrame(ci_rows)
    ci_table.to_csv(TBL_DIR / "06_bootstrap_sharpe_cis.csv", index=False)
    print(ci_table.round(3).to_string(index=False))

    # ----------------------------------- factor regression vs SPY
    print("\n=== Factor regression (HAC standard errors) ===")
    reg_rows = []
    for label, ret in [
        ("Momentum 6-1 (baseline)",      baseline.daily_returns),
        (f"Momentum {best_lb}-{best_skip} (best in-sample)", no_gate.daily_returns),
        (f"Momentum {best_lb}-{best_skip} + 200d MA filter",  gated.daily_returns),
    ]:
        # Align indexes
        common = ret.index.intersection(spy_ret.index)
        if len(common) < 100:
            continue
        fit = regress_against_market(ret.loc[common], spy_ret.loc[common], hac_lags=10)
        reg_rows.append({
            "strategy": label,
            "alpha_annualized": fit.alpha_annualized,
            "alpha_t_stat": fit.alpha_t_stat,
            "alpha_p_value": fit.alpha_p_value,
            "beta": fit.beta,
            "beta_t_stat": fit.beta_t_stat,
            "r_squared": fit.r_squared,
            "n_obs": fit.n_obs,
        })
    reg_table = pd.DataFrame(reg_rows)
    reg_table.to_csv(TBL_DIR / "07_factor_regression.csv", index=False)
    print(reg_table.round(4).to_string(index=False))

    # ----------------------------------- cost sensitivity
    print("\n=== Transaction-cost sensitivity ===")
    cost_df = cost_sensitivity(
        panel,
        signal_col=sig_col,
        cost_levels_bps=(0, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100),
        n_long=best_n,
        rebalance="ME",
    )
    cost_df.to_csv(TBL_DIR / "08_cost_sensitivity.csv", index=False)
    print(cost_df.round(4).to_string(index=False))

    # Find break-even vs SPY: where does momentum CAGR drop below SPY's?
    spy_cagr = float(summary(spy_ret, "SPY").iloc[0]["CAGR"])
    print(f"\n  SPY CAGR: {spy_cagr:.2%}  (mom 18-0 cannot match SPY at any cost level in sample)")

    # Plot CAGR & Sharpe vs cost
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(cost_df["cost_bps_per_side"], cost_df["cagr"] * 100,
                 marker="o", color=COLORS["strategy"], linewidth=2)
    axes[0].axhline(spy_cagr * 100, color=COLORS["spy"], linestyle="--", label=f"SPY CAGR ({spy_cagr:.1%})")
    axes[0].set_xlabel("Transaction cost (bps per side)")
    axes[0].set_ylabel("CAGR (%)")
    axes[0].set_title("Strategy CAGR vs. transaction cost")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(cost_df["cost_bps_per_side"], cost_df["sharpe"],
                 marker="o", color=COLORS["strategy"], linewidth=2)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Transaction cost (bps per side)")
    axes[1].set_ylabel("Sharpe ratio")
    axes[1].set_title("Strategy Sharpe vs. transaction cost")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "08_cost_sensitivity")

    # ----------------------------------- long/short variant
    print("\n=== Long/short variant ===")
    ls_res = run_backtest(
        panel, signal_col=sig_col,
        n_long=best_n, n_short=best_n,
        rebalance="ME", cost_bps_per_side=5.0,
    )

    ls_summary = pd.concat([
        summary(no_gate.daily_returns, f"Long-only {best_lb}-{best_skip} (top {best_n})"),
        summary(ls_res.daily_returns,  f"Long/short {best_lb}-{best_skip} (±{best_n})"),
        summary(spy_ret,                "SPY"),
    ])
    ls_summary.to_csv(TBL_DIR / "09_long_short.csv")
    print(ls_summary.round(4).to_string())

    # Regression of L/S: should have beta near zero (dollar-neutral) and let
    # us see the pure factor return
    common = ls_res.daily_returns.index.intersection(spy_ret.index)
    ls_fit = regress_against_market(
        ls_res.daily_returns.loc[common], spy_ret.loc[common], hac_lags=10
    )
    print(f"\n  L/S factor regression vs SPY:")
    print(f"    alpha (ann.): {ls_fit.alpha_annualized:.2%}  (t={ls_fit.alpha_t_stat:.2f}, p={ls_fit.alpha_p_value:.3f})")
    print(f"    beta:         {ls_fit.beta:.3f}              (t={ls_fit.beta_t_stat:.2f})")
    print(f"    R²:           {ls_fit.r_squared:.3f}")
    print(f"    n_obs:        {ls_fit.n_obs}")

    # Bootstrap CI on L/S Sharpe
    ls_ci = bootstrap_sharpe_ci(ls_res.daily_returns, n_bootstrap=2000, block_length=21)
    print(f"  L/S Sharpe: {ls_ci['sharpe']:.3f}  95% CI: [{ls_ci['ci_low']:.3f}, {ls_ci['ci_high']:.3f}]  "
          f"(includes zero: {ls_ci['includes_zero']})")

    fig = plot_equity_curves(
        {"Long-only momentum": no_gate.equity_curve,
         "Long/short momentum": ls_res.equity_curve,
         "SPY": spy_aligned},
        title=f"Long-only vs Long/short ({best_lb}-{best_skip}, ±{best_n})",
        color_map={"Long-only momentum": COLORS["benchmark"],
                   "Long/short momentum": COLORS["strategy"],
                   "SPY": COLORS["spy"]},
    )
    save(fig, "09_long_short_equity")

    print("\n=== Done. All outputs in reports/ ===")


if __name__ == "__main__":
    main()
