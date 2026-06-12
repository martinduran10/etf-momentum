"""Divergence filter (Phase 3 overlay).

Phase 3 stacks a divergence-based cash filter on top of the main line — Phase 1
base returns masked by the Phase 2 market-overbought filter. It is a purely
additive overlay: Phase 1 (``stack_backtest``) and Phase 2 (``overbought_filter``)
are consumed read-only and never modified.

An exogenous, precomputed divergence percentile (``data/divergence_percentile.csv``
— the spread between the top-5 and bottom-5 ETFs on the momentum signal,
expressed as a fraction in ``[0, 1]``) drives a market-level IN/OUT state
machine. The machine emits a boolean cash mask over the strategy's trading-day
calendar; Phase 3 sits in cash whenever *either* the Phase 2 mask *or* the
divergence mask says cash.

State machine (all decisions for day ``t`` use information through the close of
day ``t-1`` only — T+1 timing, no look-ahead):

* **Exit (IN → OUT)** is *edge-triggered*: it fires when the percentile crosses
  up through the threshold (``pctl[t-1] >= threshold`` and ``pctl[t-2] <
  threshold``). Merely remaining at/above the threshold does not re-fire it.
* **Re-entry (OUT → IN)** fires when *either* the percentile has normalized
  (``pctl[t-1] < threshold``) *or* the Phase 1 unfiltered cumulative curve has
  corrected at least ``reentry_drop`` points from its peak since the exit.
* **Re-arm**: because the exit needs a fresh upward cross (a prior ``< threshold``
  day), an 8%-driven re-entry while the percentile is still elevated does not
  immediately re-exit — the cross condition is self-arming, so no extra state is
  needed.

Everything stays arithmetic (no compounding): ``base_curve`` is the running sum
of Phase 1 daily returns expressed in **percentage points**, so the default
``reentry_drop = 8.0`` means an 8-percentage-point correction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR
from .stack_backtest import HEADLINE_START

#: Divergence percentile at/above which the market is "overbought-divergent".
DIVERGENCE_THRESHOLD = 0.85

#: Peak-to-trough correction (percentage points) of the Phase 1 base curve that
#: forces a re-entry while the percentile is still elevated.
REENTRY_DROP = 8.0


def load_divergence_signal(path: Path | str | None = None) -> pd.Series:
    """Load the precomputed divergence-percentile signal.

    Parameters
    ----------
    path : Path or str, optional
        Path to ``divergence_percentile.csv``. Defaults to
        ``data/divergence_percentile.csv``.

    Returns
    -------
    pandas.Series
        ``divergence_pctl`` (fraction in ``[0, 1]``) indexed by date, sorted
        ascending.
    """
    if path is None:
        path = DATA_DIR / "divergence_percentile.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()["divergence_pctl"].astype(float)


def build_divergence_cash_mask(
    pctl: pd.Series,
    base_curve: pd.Series,
    trading_days: pd.DatetimeIndex,
    threshold: float = DIVERGENCE_THRESHOLD,
    reentry_drop: float = REENTRY_DROP,
    start: int | None = None,
) -> pd.Series:
    """Run the divergence IN/OUT state machine and return its daily cash mask.

    Parameters
    ----------
    pctl : pandas.Series
        Divergence percentile indexed by date. Reindexed onto ``trading_days``
        with forward-fill ("prior available signal date" semantics).
    base_curve : pandas.Series
        Phase 1 *unfiltered* cumulative arithmetic return in **percentage
        points** (running sum of daily returns times 100), indexed over
        ``trading_days``. Used only for the ``reentry_drop`` correction test.
    trading_days : pandas.DatetimeIndex
        The strategy's full trading-day calendar. Lookbacks (``t-1``, ``t-2``)
        are positional in this calendar, so the full calendar must be passed —
        the first strategy day's lookbacks precede the headline window.
    threshold : float, optional
        Exit threshold (default :data:`DIVERGENCE_THRESHOLD`). Exit uses
        ``>= threshold`` (inclusive); signal-path re-entry uses ``< threshold``
        (strict). ``threshold`` exactly counts as overbought-divergent.
    reentry_drop : float, optional
        Correction in base-curve points that forces a re-entry (default
        :data:`REENTRY_DROP`, i.e. 8 percentage points).
    start : int, optional
        Positional index of the first strategy trading day in ``trading_days``.
        Defaults to the position of :data:`~src.stack_backtest.HEADLINE_START`.
        Days before ``start`` are IN (mask ``False``). Exposed for deterministic
        unit testing on synthetic calendars.

    Returns
    -------
    pandas.Series of bool
        ``True`` where the day is cash (OUT), indexed by ``trading_days``.

    Notes
    -----
    No look-ahead: ``mask[t]`` reads ``pctl`` and ``base_curve`` only at ``t-1``
    and earlier (the exit's upward cross inspects ``t-1`` and ``t-2``).
    """
    p = pctl.reindex(trading_days).ffill().to_numpy()
    bc = base_curve.reindex(trading_days).to_numpy()
    n = len(trading_days)

    if start is None:
        start = int(trading_days.searchsorted(pd.Timestamp(HEADLINE_START)))

    cash = np.zeros(n, dtype=bool)
    if start < 1 or start >= n:
        # No room for the t-1 init lookback (or start past the calendar):
        # nothing to do — the strategy is IN throughout the available range.
        return pd.Series(cash, index=trading_days)

    # Initialization at the first strategy trading day: level check on the prior
    # signal day. IN unless the percentile is already at/above the threshold.
    state_in = p[start - 1] < threshold
    peak = bc[start - 1] if not state_in else np.nan
    cash[start] = not state_in

    for i in range(start + 1, n):
        if state_in:
            crossed_up = p[i - 1] >= threshold and p[i - 2] < threshold
            if crossed_up:
                state_in = False
                peak = bc[i - 1]  # exit-trigger day (t-1 cross), inclusive
                cash[i] = True
            # else: stay IN, cash[i] already False
        else:
            if bc[i - 1] > peak:
                peak = bc[i - 1]
            reenter = (p[i - 1] < threshold) or (
                bc[i - 1] <= peak - reentry_drop
            )
            if reenter:
                state_in = True
            else:
                cash[i] = True

    return pd.Series(cash, index=trading_days)


def apply_phase3(
    phase1_returns: pd.Series,
    phase2_cash_mask: pd.Series,
    divergence_cash_mask: pd.Series,
) -> pd.Series:
    """Stack the divergence mask on the Phase 2 mask and filter Phase 1 returns.

    Parameters
    ----------
    phase1_returns : pandas.Series
        Phase 1 combined daily returns (the series Phase 3 masks).
    phase2_cash_mask : pandas.Series of bool
        The Phase 2 market-overbought cash mask.
    divergence_cash_mask : pandas.Series of bool
        The divergence cash mask from :func:`build_divergence_cash_mask`.

    Returns
    -------
    pandas.Series
        ``0.0`` on any day either mask flags as cash, else the Phase 1 return.
        Same index as ``phase1_returns``.
    """
    idx = phase1_returns.index
    m2 = phase2_cash_mask.reindex(idx, fill_value=False)
    md = divergence_cash_mask.reindex(idx, fill_value=False)
    cash = m2 | md
    return phase1_returns.where(~cash, 0.0)
