"""Tests for the Phase 2b individual-ETF overbought filter.

These pin mechanics on synthetic data — the overbought formula and its
boundaries, the analyze gate, daily selection / substitution / cash, the
no-timer swap-back, and the no-look-ahead roster lag — not production numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import load_closes
from src.individual_overbought import (
    SMA_LONG_MULTIPLIER,
    SMA_LONG_WINDOW,
    SMA_SHORT_MULTIPLIER,
    SMA_SHORT_WINDOW,
    compute_eligibility,
    compute_overbought_flags,
    run_threshold_sweep,
    select_daily_roster,
    sub_returns_from_selection,
)
from src.overbought_filter import build_cash_mask, load_overbought_signal


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


# --------------------------------------------------------------------------- #
# Overbought formula and its boundaries
# --------------------------------------------------------------------------- #


def test_constant_price_never_overbought() -> None:
    """A flat price sits below both 1.05x and 1.07x of its own SMA."""
    closes = pd.DataFrame({"a": [100.0] * 60}, index=_days(60))
    flags = compute_overbought_flags(closes)
    assert not flags["a"].any()


def test_short_sma_boundary_is_inclusive() -> None:
    """close == 1.05 * SMA20 (exactly) is overbought; just below is not.

    The 20-day window is eighteen 20s + one 19 + the test close. With close=21
    the window sums to 400 -> SMA20 = 20.0 exactly, and 1.05 * 20.0 == 21.0 as
    IEEE doubles, so the inclusive ``>=`` must flag the final day. The series is
    only 20 long, so SMA50 is NaN and cannot interfere.
    """
    window_prefix = [20.0] * 18 + [19.0]  # 19 prior days, sum 379

    hot = pd.DataFrame({"a": window_prefix + [21.0]}, index=_days(20))
    assert compute_overbought_flags(hot)["a"].iloc[-1]

    # Raise the prior sum so the same close lands strictly below the boundary.
    cool = pd.DataFrame({"a": [20.0] * 19 + [21.0]}, index=_days(20))
    assert not compute_overbought_flags(cool)["a"].iloc[-1]


def test_long_sma_condition_fires_independently_or_logic() -> None:
    """The 1.07x-SMA50 leg can flag a name on its own (OR logic).

    An uptrend lifts SMA20 well above the close, so the short leg stays quiet;
    the close is engineered to cross 1.07 * SMA50 only.
    """
    prices = [90.0] * 30 + [110.0] * 19  # 49 days, recent ramp up
    hot = pd.DataFrame({"a": prices + [106.0]}, index=_days(50))
    long_hot = pd.DataFrame({"a": prices + [103.0]}, index=_days(50))

    flags_hot = compute_overbought_flags(hot)
    closes_hot = hot["a"]
    sma20 = closes_hot.rolling(SMA_SHORT_WINDOW).mean().iloc[-1]
    sma50 = closes_hot.rolling(SMA_LONG_WINDOW).mean().iloc[-1]
    # Short leg quiet, long leg fires -> overbought via OR.
    assert closes_hot.iloc[-1] < SMA_SHORT_MULTIPLIER * sma20
    assert closes_hot.iloc[-1] >= SMA_LONG_MULTIPLIER * sma50
    assert flags_hot["a"].iloc[-1]

    # A lower close clears both legs.
    assert not compute_overbought_flags(long_hot)["a"].iloc[-1]


def test_overbought_matches_independent_recomputation() -> None:
    """On a random series the flags equal an independent SMA recomputation."""
    rng = np.random.default_rng(7)
    idx = _days(120)
    closes = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(120, 4)), axis=0)),
        index=idx,
        columns=["a", "b", "c", "d"],
    )
    flags = compute_overbought_flags(closes)

    sma_short = closes.rolling(SMA_SHORT_WINDOW).mean()
    sma_long = closes.rolling(SMA_LONG_WINDOW).mean()
    expected = (closes >= SMA_SHORT_MULTIPLIER * sma_short) | (
        closes >= SMA_LONG_MULTIPLIER * sma_long
    )
    pd.testing.assert_frame_equal(flags, expected)


def test_short_window_etf_not_flagged() -> None:
    """An ETF without a full 20-day window is never flagged (NaN -> False)."""
    closes = pd.DataFrame(
        {"a": [10.0, 50.0, 1.0, 99.0, 2.0]}, index=_days(5)
    )
    flags = compute_overbought_flags(closes)
    assert not flags["a"].any()


# --------------------------------------------------------------------------- #
# Analyze gate
# --------------------------------------------------------------------------- #


def test_screen_active_only_on_analyze_one_days() -> None:
    """An overbought name is excluded only on analyze==1 days."""
    idx = _days(2)
    slow = pd.DataFrame({"a": [1.0, 1.0]}, index=idx)  # positive both days
    overbought = pd.DataFrame({"a": [True, True]}, index=idx)
    analyze = pd.Series([1, 0], index=idx)

    eligible = compute_eligibility(slow, overbought, analyze, screen=True)
    assert not eligible["a"].iloc[0]  # analyze==1 -> screened out
    assert eligible["a"].iloc[1]  # analyze==0 -> screen off, eligible


def test_screen_off_ignores_overbought() -> None:
    """With screen=False, eligibility is purely slow_signal > 0."""
    idx = _days(2)
    slow = pd.DataFrame({"a": [1.0, -1.0]}, index=idx)
    overbought = pd.DataFrame({"a": [True, True]}, index=idx)
    analyze = pd.Series([1, 1], index=idx)

    eligible = compute_eligibility(slow, overbought, analyze, screen=False)
    assert eligible["a"].iloc[0]  # positive signal, screen ignored
    assert not eligible["a"].iloc[1]  # non-positive signal


def test_non_positive_signal_never_eligible() -> None:
    """Zero or negative slow signal is never eligible, screen or not."""
    idx = _days(1)
    slow = pd.DataFrame({"a": [0.0], "b": [-0.5]}, index=idx)
    overbought = pd.DataFrame({"a": [False], "b": [False]}, index=idx)
    analyze = pd.Series([0], index=idx)
    eligible = compute_eligibility(slow, overbought, analyze, screen=True)
    assert not eligible.iloc[0].any()


# --------------------------------------------------------------------------- #
# Daily selection, substitution, cash
# --------------------------------------------------------------------------- #


def test_substitution_fills_from_next_ranked_name() -> None:
    """A screened-out top name yields its slot to the next eligible name."""
    idx = _days(1)
    cols = ["a", "b", "c", "d", "e", "f"]
    # Descending slow signal a>b>c>d>e>f, all positive.
    slow = pd.DataFrame([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], index=idx, columns=cols)
    overbought = pd.DataFrame(
        [[True, False, False, False, False, False]], index=idx, columns=cols
    )
    analyze = pd.Series([1], index=idx)

    eligible = compute_eligibility(slow, overbought, analyze, screen=True)
    selected = select_daily_roster(slow, eligible, start_idx=0, n_slots=5)
    chosen = set(selected.columns[selected.iloc[0].to_numpy()])
    # 'a' is screened out; the top-5 eligible are b,c,d,e,f.
    assert chosen == {"b", "c", "d", "e", "f"}


def test_fewer_than_five_eligible_leaves_cash_slots() -> None:
    """With <5 eligible names the divisor stays 5, so unfilled slots are cash."""
    idx = _days(2)
    cols = ["a", "b", "c"]
    slow = pd.DataFrame(
        [[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], index=idx, columns=cols
    )
    overbought = pd.DataFrame(False, index=idx, columns=cols)
    analyze = pd.Series([0, 0], index=idx)

    eligible = compute_eligibility(slow, overbought, analyze, screen=True)
    selected = select_daily_roster(slow, eligible, start_idx=0, n_slots=5)
    assert int(selected.iloc[0].sum()) == 3  # only three names available

    # Each name returns 10% on day 1; sub return = (0.10 * 3) / 5 = 0.06.
    simple = pd.DataFrame(
        [[np.nan, np.nan, np.nan], [0.10, 0.10, 0.10]], index=idx, columns=cols
    )
    sub = sub_returns_from_selection(simple, selected, roster_lag=1, n_slots=5)
    assert sub.iloc[1] == pytest.approx(0.06)


def test_cooled_off_name_reclaims_slot_next_day() -> None:
    """No timer: a name excluded while hot is eligible again once it cools."""
    idx = _days(2)
    cols = ["a", "b"]
    slow = pd.DataFrame([[2.0, 1.0], [2.0, 1.0]], index=idx, columns=cols)
    # 'a' overbought day 0, cool day 1.
    overbought = pd.DataFrame(
        [[True, False], [False, False]], index=idx, columns=cols
    )
    analyze = pd.Series([1, 1], index=idx)

    eligible = compute_eligibility(slow, overbought, analyze, screen=True)
    selected = select_daily_roster(slow, eligible, start_idx=0, n_slots=5)
    assert not selected["a"].iloc[0]  # excluded while overbought
    assert selected["a"].iloc[1]  # eligible again immediately when cool


# --------------------------------------------------------------------------- #
# Cooldown timer (cooldown_days) and the screen-off attribution path
# --------------------------------------------------------------------------- #


def test_cooldown_zero_is_no_timer() -> None:
    """cooldown_days=0 reproduces the no-timer behavior: eligible next day."""
    idx = _days(3)
    slow = pd.DataFrame({"a": [1.0, 1.0, 1.0]}, index=idx)
    overbought = pd.DataFrame({"a": [True, False, False]}, index=idx)
    analyze = pd.Series([1, 1, 1], index=idx)

    eligible = compute_eligibility(
        slow, overbought, analyze, screen=True, cooldown_days=0
    )
    assert not eligible["a"].iloc[0]  # flagged on day 0
    assert eligible["a"].iloc[1]  # cooled -> eligible immediately, no timer


def test_cooldown_five_bars_exactly_five_days_after_a_flag() -> None:
    """A name flagged only on day 0 stays barred days 1-5, eligible on day 6.

    The flag day itself is excluded by the same-day screen; the cooldown adds
    the next five trading days even though the name has already cooled.
    """
    idx = _days(8)
    slow = pd.DataFrame({"a": [1.0] * 8}, index=idx)
    overbought = pd.DataFrame({"a": [True] + [False] * 7}, index=idx)
    analyze = pd.Series([1] * 8, index=idx)

    eligible = compute_eligibility(
        slow, overbought, analyze, screen=True, cooldown_days=5
    )
    for d in range(6):  # day 0 (screen) + days 1-5 (cooldown)
        assert not eligible["a"].iloc[d], d
    assert eligible["a"].iloc[6]  # cooldown expired
    assert eligible["a"].iloc[7]


def test_cooldown_fresh_flag_resets_the_clock() -> None:
    """A second flag while barred restarts the five-day count (windows merge)."""
    idx = _days(10)
    slow = pd.DataFrame({"a": [1.0] * 10}, index=idx)
    # Hits on day 0 and day 2.
    overbought = pd.DataFrame(
        {"a": [True, False, True] + [False] * 7}, index=idx
    )
    analyze = pd.Series([1] * 10, index=idx)

    eligible = compute_eligibility(
        slow, overbought, analyze, screen=True, cooldown_days=5
    )
    # Latest hit is day 2 -> barred through day 7, eligible on day 8.
    for d in range(8):
        assert not eligible["a"].iloc[d], d
    assert eligible["a"].iloc[8]


def test_cooldown_bar_persists_across_analyze_zero_day() -> None:
    """An existing bar keeps counting on analyze==0 days, and an overbought
    reading on such a day starts no new bar."""
    idx = _days(8)
    slow = pd.DataFrame({"a": [1.0] * 8}, index=idx)
    # Overbought on day 0 (analyze on) and day 3 (analyze OFF).
    overbought = pd.DataFrame(
        {"a": [True, False, False, True, False, False, False, False]}, index=idx
    )
    analyze = pd.Series([1, 1, 1, 0, 1, 1, 1, 1], index=idx)

    eligible = compute_eligibility(
        slow, overbought, analyze, screen=True, cooldown_days=5
    )
    # Only day 0 is a hit (day 3's overbought is ignored: analyze==0).
    assert not eligible["a"].iloc[3]  # bar from day 0 persists across analyze==0
    assert eligible["a"].iloc[6]  # day-3 overbought did NOT extend the clock


def test_screen_off_ignores_overbought_and_cooldown() -> None:
    """The attribution path (screen off) excludes nothing for overbought reasons,
    even with a cooldown set."""
    idx = _days(3)
    slow = pd.DataFrame({"a": [1.0, -1.0, 2.0]}, index=idx)
    overbought = pd.DataFrame({"a": [True, True, True]}, index=idx)
    analyze = pd.Series([1, 1, 1], index=idx)

    eligible = compute_eligibility(
        slow, overbought, analyze, screen=False, cooldown_days=5
    )
    pd.testing.assert_frame_equal(eligible, slow > 0)


# --------------------------------------------------------------------------- #
# Threshold robustness sweep harness
# --------------------------------------------------------------------------- #


def test_threshold_sweep_runs_and_canonical_pair_reproduces_phase2b() -> None:
    """The sweep harness runs, and the 5%/7% row reproduces the shipped Phase 2b
    headline (124.30% / 0.86 / -24.04%) as a regression guard."""
    closes = load_closes()
    cash_mask = build_cash_mask(load_overbought_signal(), closes.index)

    sweep = run_threshold_sweep(closes, cash_mask, grid=[(0.05, 0.07)])

    assert list(sweep.columns) == [
        "sma20_pct",
        "sma50_pct",
        "total_return",
        "sharpe",
        "max_dd",
    ]
    assert len(sweep) == 1
    row = sweep.iloc[0]
    assert (row["sma20_pct"], row["sma50_pct"]) == (0.05, 0.07)
    # Canonical Phase 2b numbers (full precision, frozen at the prior commit).
    assert row["total_return"] == pytest.approx(1.243019253941057)
    assert row["sharpe"] == pytest.approx(0.8633661482172864)
    assert row["max_dd"] == pytest.approx(-0.24039697479841388)


# --------------------------------------------------------------------------- #
# No-look-ahead roster lag
# --------------------------------------------------------------------------- #


def test_roster_lag_uses_prior_day_selection() -> None:
    """Day t's return accrues the roster selected at t-1, not t."""
    idx = _days(3)
    cols = ["a"]
    selected = pd.DataFrame([[False], [True], [False]], index=idx, columns=cols)
    # 'a' returns 10% on every day.
    simple = pd.DataFrame([[np.nan], [0.10], [0.10]], index=idx, columns=cols)

    sub = sub_returns_from_selection(simple, selected, roster_lag=1, n_slots=5)
    # Selected at position 1 -> accrues at position 2 only.
    assert sub.iloc[0] == 0.0
    assert sub.iloc[1] == 0.0  # nothing held yet on day 1
    assert sub.iloc[2] == 0.10 / 5  # day-2 return from the day-1 roster


def test_selection_before_start_idx_is_empty() -> None:
    """No roster is selected before the sub-strategy's start position."""
    idx = _days(4)
    cols = ["a", "b"]
    slow = pd.DataFrame(
        [[2.0, 1.0]] * 4, index=idx, columns=cols
    )
    overbought = pd.DataFrame(False, index=idx, columns=cols)
    analyze = pd.Series([0, 0, 0, 0], index=idx)

    eligible = compute_eligibility(slow, overbought, analyze, screen=True)
    selected = select_daily_roster(slow, eligible, start_idx=2, n_slots=5)
    assert not selected.iloc[0].any()
    assert not selected.iloc[1].any()
    assert selected.iloc[2].any()
    assert selected.iloc[3].any()
