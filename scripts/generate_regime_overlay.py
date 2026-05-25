"""Phase 1 driver — fit the HMM regime detector and emit overlay artifacts.

Reproduces every regime-detection figure and table in RESULTS_v2.md.
Outputs land in ``reports/figures/`` and ``reports/tables/``.

Usage:
    python scripts/generate_regime_overlay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from src.data import load_panel
from src.regime_hmm import (
    compute_regime_features,
    fit_hmm_regimes,
    expand_regimes_to_daily,
    regime_summary_stats,
)
from src.visualization import (
    plot_regime_overlay,
    plot_transition_matrix,
    COLORS,
)


FIG_DIR = REPO_ROOT / "reports" / "figures"
TBL_DIR = REPO_ROOT / "reports" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)


REGIME_LABELS = {0: "Bear", 1: "Neutral", 2: "Bull"}
REGIME_COLORS = {
    0: COLORS["spy"],         # muted red
    1: COLORS["benchmark"],   # neutral grey
    2: COLORS["improved"],    # forest green
}


def save(fig, name: str) -> None:
    """Save a figure as PNG and close it to free memory."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved {path.relative_to(REPO_ROOT)}")


def main() -> None:
    print("\n=== Loading data ===")
    panel = load_panel()
    print(f"  Panel: {len(panel):,} rows, "
          f"{panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    print("\n=== Computing weekly features ===")
    features = compute_regime_features(panel, ticker="spy", vol_window=4)
    print(f"  Features: {features.shape[0]} weeks, "
          f"{features.index.min().date()} → {features.index.max().date()}")

    print("\n=== Fitting 3-state Gaussian HMM ===")
    regimes, model = fit_hmm_regimes(features, n_regimes=3, seed=42)
    print(f"  Converged after {model.monitor_.iter} EM iterations")
    print("  Transition matrix (rows = from, cols = to):")
    for i, row in enumerate(model.transmat_):
        label = REGIME_LABELS[i]
        probs = "  ".join(f"{p:.3f}" for p in row)
        print(f"    {label:7s} {probs}")

    print("\n=== Per-regime summary ===")
    summary = regime_summary_stats(features, regimes, daily_panel=panel, ticker="spy")
    summary.insert(0, "name", [REGIME_LABELS[r] for r in summary.index])
    summary.to_csv(TBL_DIR / "regime_hmm_summary.csv")
    print(summary.round(4).to_string())

    print("\n=== Building overlay chart ===")
    spy_close = (
        panel.loc[panel["ticker"] == "spy", ["date", "close"]]
        .set_index("date")
        .sort_index()["close"]
    )
    spy_close.name = "SPY close"
    daily_regimes = expand_regimes_to_daily(regimes, spy_close.index)
    fig = plot_regime_overlay(
        price=spy_close,
        daily_regimes=daily_regimes,
        regime_labels=REGIME_LABELS,
        regime_colors=REGIME_COLORS,
        title="SPY with HMM regime overlay (3 states, weekly fit)",
    )
    save(fig, "regime_hmm_overlay")

    print("\n=== Building transition matrix heatmap ===")
    fig = plot_transition_matrix(
        transmat=model.transmat_,
        labels=[REGIME_LABELS[i] for i in range(model.n_components)],
        title="HMM regime transition matrix (weekly)",
    )
    save(fig, "regime_hmm_transition_matrix")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
