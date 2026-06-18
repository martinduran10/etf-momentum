"""Regenerate ``data/divergence_percentile.csv`` from the raw closes.

Replaces the previously exogenous divergence input with a reproducible,
in-repo artifact. Run after any change to the closes or the signal definition.

Usage
-----
    python scripts/build_divergence_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import DATA_DIR, load_closes  # noqa: E402
from src.divergence_signal import build_divergence_signal  # noqa: E402


def main() -> None:
    closes = load_closes()
    df = build_divergence_signal(closes)

    out_path = DATA_DIR / "divergence_percentile.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print(
        f"  date range : {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}"
    )
    print(
        f"  pctl range : {df['divergence_pctl'].min():.4f}"
        f" -> {df['divergence_pctl'].max():.4f}"
    )


if __name__ == "__main__":
    main()
