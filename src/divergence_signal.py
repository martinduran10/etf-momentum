"""Divergence signal construction (reproducible Phase 3 input).

Phase 3 consumes a divergence percentile that previously arrived as an opaque,
precomputed CSV (``data/divergence_percentile.csv``). This module recomputes it
from the raw closes already in the repo, turning that input into a generated,
auditable artifact.

Definition (per trading day ``t``):

1. ``slow_signal`` is the Phase 1 risk-adjusted-momentum signal at the **300-day
   lookback** (sub-strategy C's lookback), computed per ETF from the raw closes
   via :mod:`src.signals`.
2. The daily **divergence spread** is the sum of the five largest ``slow_signal``
   values across the 43 ETFs minus the sum of the five smallest, on day ``t`` —
   a measure of how far the best- and worst-momentum names have separated.
3. The **divergence percentile** is the expanding empirical rank of day ``t``'s
   spread within every spread observed up to **and including** ``t`` (a fraction
   in ``(0, 1]``). Only past-and-present spreads enter the rank, so the value is
   knowable at ``t``'s close; Phase 3's state machine then reads it at ``t-1``
   for its T+1, no-look-ahead decision.

The output column (``divergence_pctl``) and file layout are drop-in compatible
with the loader in :mod:`src.divergence_filter`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import compute_risk_adj_return, compute_slow_signal

#: Slow-signal lookback (trading days) feeding the divergence spread — matches
#: sub-strategy C's lookback in :data:`src.stack_backtest.SUB_STRATEGIES`.
DIVERGENCE_LOOKBACK = 300

#: Number of names taken from each tail when forming the top-minus-bottom spread.
N_EXTREME = 5


def compute_divergence_spread(
    slow_signal: pd.DataFrame, n_extreme: int = N_EXTREME
) -> pd.Series:
    """Daily top-``n_extreme`` minus bottom-``n_extreme`` slow-signal spread.

    For each trading day the ETFs are ranked by ``slow_signal``; the spread is
    the sum of the ``n_extreme`` largest values minus the sum of the
    ``n_extreme`` smallest. A day with fewer than ``2 * n_extreme`` defined
    signals (i.e. the warm-up period before the lookback fills) yields ``NaN``.

    Parameters
    ----------
    slow_signal : pandas.DataFrame
        Slow signal per ETF (dates x tickers), one lookback.
    n_extreme : int, optional
        Names taken from each tail (default :data:`N_EXTREME`).

    Returns
    -------
    pandas.Series
        The daily divergence spread indexed by date; ``NaN`` before the lookback
        fills.

    Notes
    -----
    ``Series.nlargest`` / ``nsmallest`` ignore ``NaN``, so partially-defined days
    are handled by the explicit ``2 * n_extreme`` validity guard rather than by
    silent fill.
    """

    def _row_spread(row: pd.Series) -> float:
        if int(row.notna().sum()) < 2 * n_extreme:
            return np.nan
        return float(row.nlargest(n_extreme).sum() - row.nsmallest(n_extreme).sum())

    return slow_signal.apply(_row_spread, axis=1)


def compute_divergence_percentile(spread: pd.Series) -> pd.Series:
    """Expanding, inclusive empirical percentile of the divergence spread.

    The percentile on day ``t`` is the fraction of spreads observed up to and
    including ``t`` that are ``<=`` the day-``t`` spread. Because the rank uses
    only past-and-present values, appending later data never changes an earlier
    percentile — the no-look-ahead property pinned in the tests.

    Parameters
    ----------
    spread : pandas.Series
        The daily divergence spread (may contain leading ``NaN``).

    Returns
    -------
    pandas.Series
        Percentile in ``(0, 1]`` aligned to ``spread``'s index; ``NaN`` wherever
        the spread itself is ``NaN``.
    """
    valid = spread.dropna()
    values = valid.to_numpy()
    out = np.empty(len(values), dtype=float)
    for i in range(len(values)):
        window = values[: i + 1]
        out[i] = float((window <= values[i]).mean())
    return pd.Series(out, index=valid.index).reindex(spread.index)


def build_divergence_signal(
    closes: pd.DataFrame,
    lookback: int = DIVERGENCE_LOOKBACK,
    n_extreme: int = N_EXTREME,
) -> pd.DataFrame:
    """Build the full ``date, divergence_pctl`` table from raw closes.

    Wires the pieces together: closes -> 300-day slow signal -> daily spread ->
    expanding percentile. Leading rows with an undefined spread (the lookback
    warm-up) are dropped so the output carries no ``NaN`` — matching the
    cleanliness the :mod:`src.divergence_filter` loader expects.

    Parameters
    ----------
    closes : pandas.DataFrame
        Close prices indexed by date, one column per ETF.
    lookback : int, optional
        Slow-signal lookback (default :data:`DIVERGENCE_LOOKBACK`).
    n_extreme : int, optional
        Tail size for the spread (default :data:`N_EXTREME`).

    Returns
    -------
    pandas.DataFrame
        Two columns, ``date`` and ``divergence_pctl``, sorted ascending, with no
        missing values — ready to write to ``data/divergence_percentile.csv``.
    """
    risk_adj = compute_risk_adj_return(closes)
    slow_signal = compute_slow_signal(risk_adj, lookback)
    spread = compute_divergence_spread(slow_signal, n_extreme=n_extreme)
    pctl = compute_divergence_percentile(spread).dropna().rename("divergence_pctl")
    pctl.index.name = "date"
    return pctl.reset_index()
