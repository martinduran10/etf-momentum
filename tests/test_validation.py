"""Frozen-snapshot regression for the Phase 1 and Phase 2 daily-return series.

The project no longer validates against the investor deck's published numbers —
phases are reported empirically. This module instead pins the Phase 1 (Stack
Portfolio) and Phase 2 (overbought-filtered) daily-return series byte-for-byte,
enforcing the additive-overlay principle: a downstream phase (or any other edit)
must never perturb an earlier one.

Snapshots live in ``tests/fixtures/*.parquet`` (exact float64 round-trip). A
fixture is created on first run and compared exactly on every run thereafter;
commit the fixtures so the guard is active in CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import load_closes
from src.overbought_filter import (
    apply_filter,
    build_cash_mask,
    load_overbought_signal,
)
from src.stack_backtest import run_stack_portfolio

FIXTURES = Path(__file__).parent / "fixtures"


def _check_snapshot(series: pd.Series, name: str) -> None:
    """Compare ``series`` to its pinned parquet snapshot, bootstrapping if absent.

    The index is normalized to nanosecond resolution before comparison so the
    guard is robust to pandas' datetime-resolution differences across versions
    (e.g. read_csv yielding datetime64[us] on pandas 3.x vs [ns] in the committed
    fixtures). Only the index storage unit is harmonized; the daily-return values
    are still compared byte-for-byte (check_exact=True).
    """
    FIXTURES.mkdir(exist_ok=True)
    path = FIXTURES / f"{name}.parquet"
    current = series.rename("ret").to_frame()
    current.index = current.index.as_unit("ns")
    if not path.exists():
        current.to_parquet(path)
    expected = pd.read_parquet(path)
    expected.index = expected.index.as_unit("ns")
    pd.testing.assert_frame_equal(current, expected, check_exact=True, check_freq=False)


def _phase1_phase2_series() -> tuple[pd.Series, pd.Series]:
    closes = load_closes()
    result = run_stack_portfolio(closes)
    headline = result["headline_returns"]
    cash_mask = build_cash_mask(load_overbought_signal(), closes.index)
    filtered = apply_filter(headline, cash_mask)
    return headline, filtered


def test_phase1_daily_returns_match_snapshot() -> None:
    headline, _ = _phase1_phase2_series()
    _check_snapshot(headline, "phase1_headline_returns")


def test_phase2_daily_returns_match_snapshot() -> None:
    _, filtered = _phase1_phase2_series()
    _check_snapshot(filtered, "phase2_filtered_returns")
