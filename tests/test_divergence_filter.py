"""Tests for the Phase 3 divergence filter.

The loader is checked against the shipped signal; the state machine is pinned on
small hand-built synthetic fixtures with an explicit ``start`` so each transition
(edge-triggered exit, dual re-entry, re-arm, peak tracking, boundary, init) is
isolated. Decisions are T+1 (info through ``t-1`` only).
"""

from __future__ import annotations

import pandas as pd

from src.divergence_filter import (
    apply_phase3,
    build_divergence_cash_mask,
    load_divergence_signal,
)


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _mask(pctl_vals, bc_vals, start=1, threshold=0.85, reentry_drop=8.0):
    idx = _days(len(pctl_vals))
    pctl = pd.Series(pctl_vals, index=idx, dtype=float)
    base_curve = pd.Series(bc_vals, index=idx, dtype=float)
    return build_divergence_cash_mask(
        pctl, base_curve, idx, threshold=threshold,
        reentry_drop=reentry_drop, start=start,
    )


# --------------------------------------------------------------------------- #
# Loader (real shipped file)
# --------------------------------------------------------------------------- #


def test_loader_shape_bounds_and_cleanliness() -> None:
    sig = load_divergence_signal()
    assert isinstance(sig.index, pd.DatetimeIndex)
    assert sig.index.is_monotonic_increasing
    assert not sig.index.has_duplicates
    assert sig.notna().all()
    assert float(sig.min()) >= 0.0 and float(sig.max()) <= 1.0
    assert len(sig) == 2764
    assert sig.index.min() == pd.Timestamp("2015-06-16")
    assert sig.index.max() == pd.Timestamp("2026-06-11")


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #


def test_cross_up_puts_strategy_in_cash_next_day_t_plus_1() -> None:
    """An upward cross through 0.85 forces cash the FOLLOWING day, not the cross
    day itself (T+1)."""
    # pctl crosses at position 3 (0.5 -> 0.9 between pos 2 and 3).
    pctl = [0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
    bc = [100.0] * 7
    mask = _mask(pctl, bc, start=1)
    assert not mask.iloc[3]  # cross day: still invested
    assert mask.iloc[4]  # T+1: cash begins


def test_eight_percent_reentry_and_rearm() -> None:
    """An 8-point correction from the peak-since-exit re-enters even while the
    percentile stays elevated, and the strategy then STAYS in (re-arm: no exit
    without a fresh cross)."""
    pctl = [0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    bc = [100.0, 100.0, 100.0, 100.0, 95.0, 91.0, 91.0, 91.0, 91.0]
    mask = _mask(pctl, bc, start=1)
    assert mask.iloc[3]  # exit (cross at pos 2 -> cash at pos 3)
    assert mask.iloc[4] and mask.iloc[5]  # still OUT
    assert not mask.iloc[6]  # bc 91 <= 100 - 8 -> 8%-path re-entry
    assert not mask.iloc[7] and not mask.iloc[8]  # re-arm: stays IN, no re-exit


def test_signal_path_reentry() -> None:
    """The percentile dropping back below 0.85 re-enters the next day."""
    pctl = [0.5, 0.5, 0.9, 0.9, 0.5, 0.5]
    bc = [100.0] * 6
    mask = _mask(pctl, bc, start=1)
    assert mask.iloc[3]  # exit
    assert mask.iloc[4]  # still OUT
    assert not mask.iloc[5]  # pctl < 0.85 -> re-entry


def test_fresh_cross_re_exits_after_reentry() -> None:
    """Positive control: after a re-entry, a genuine fresh upward cross fires a
    second exit."""
    pctl = [0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.9, 0.9]
    bc = [100.0] * 8
    mask = _mask(pctl, bc, start=1)
    assert mask.iloc[3]  # first exit
    assert not mask.iloc[5]  # signal re-entry
    assert mask.iloc[7]  # fresh cross (pos 5 <0.85 -> pos 6 >=0.85) re-exits


def test_peak_since_exit_tracks_rising_base_curve() -> None:
    """Re-entry's 8-point test is measured from the running peak AFTER the exit,
    not the exit-day level."""
    # Exit at pos 3 with base 100; base rises to 110, then falls to 95.
    pctl = [0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    bc = [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 95.0, 95.0]
    mask = _mask(pctl, bc, start=1)
    assert mask.iloc[3]  # exit, peak initialized at 100
    assert mask.iloc[7]  # still OUT just before the drop registers
    # 95 <= 110 - 8 (=102) re-enters; would NOT if peak were stuck at 100 (92).
    assert not mask.iloc[8]


def test_threshold_boundary_is_overbought_divergent_at_exactly_085() -> None:
    """0.85 exactly counts as overbought-divergent: it triggers the exit cross
    and does not satisfy the strict (< 0.85) signal re-entry."""
    pctl = [0.5, 0.5, 0.85, 0.85, 0.85]
    bc = [100.0] * 5
    mask = _mask(pctl, bc, start=1)
    assert mask.iloc[3]  # cross to exactly 0.85 -> exit
    assert mask.iloc[4]  # 0.85 is not < 0.85 -> no signal re-entry, stays OUT


def test_initialization_in_when_prior_below_threshold() -> None:
    pctl = [0.80, 0.5, 0.5]
    bc = [100.0] * 3
    mask = _mask(pctl, bc, start=1)
    assert not mask.iloc[1]  # prior pctl 0.80 < 0.85 -> starts IN


def test_initialization_out_when_prior_at_or_above_threshold() -> None:
    pctl = [0.90, 0.5, 0.5]
    bc = [100.0] * 3
    mask = _mask(pctl, bc, start=1)
    assert mask.iloc[1]  # prior pctl 0.90 >= 0.85 -> starts OUT
    assert not mask.iloc[2]  # then pctl 0.5 < 0.85 -> re-entry


def test_no_look_ahead_shift_changes_mask() -> None:
    """Shifting the percentile forward by one day changes the mask (guards
    against accidental same-day usage)."""
    idx = _days(7)
    pctl = pd.Series([0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9], index=idx, dtype=float)
    bc = pd.Series([100.0] * 7, index=idx, dtype=float)
    base = build_divergence_cash_mask(pctl, bc, idx, start=1)
    shifted = build_divergence_cash_mask(pctl.shift(1), bc, idx, start=1)
    assert not base.equals(shifted)


# --------------------------------------------------------------------------- #
# Stacking
# --------------------------------------------------------------------------- #


def test_apply_phase3_cash_is_union_of_masks() -> None:
    idx = _days(5)
    phase1 = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
    m2 = pd.Series([False, True, False, False, False], index=idx)
    md = pd.Series([False, False, True, True, False], index=idx)

    out = apply_phase3(phase1, m2, md)

    expected = pd.Series([0.01, 0.0, 0.0, 0.0, 0.05], index=idx)
    pd.testing.assert_series_equal(out, expected, check_names=False)
    # Zeroed days are exactly the union of the two cash masks.
    assert list((out == 0.0)) == list(m2 | md)
