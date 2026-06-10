"""Individual-ETF overbought filter (Phase 2b overlay).

Phase 2b sits **on top of** Phase 1's mechanics as a purely additive overlay:
no Phase 1 / Phase 2 module, test, or generated artifact is touched. It reuses
Phase 1's signal construction (:mod:`src.signals`), base returns
(:mod:`src.data`), sub-strategy configs (:mod:`src.stack_backtest`), and the
Phase 2 market cash mask (:mod:`src.overbought_filter`).

Where Phase 1 rebalances every 20 trading days, Phase 2b **rebalances daily**:
each of the four sub-strategies (A/B/C/D, lookbacks 260/280/300/320, same start
dates) re-ranks all 43 ETFs by its slow signal every day and holds the top five
*eligible* names. Eligibility is

* ``slow_signal > 0`` (never reach into non-positive names), **and**
* on days the analyze gate is active (``analyze == 1``), *not* individually
  overbought.

An ETF is individually overbought on a day when its close sits at or above
``1.05`` x its 20-day SMA **or** at or above ``1.07`` x its 50-day SMA (simple
moving averages of that ETF's own close). Either condition is enough (logical
OR). There is **no timer**: a name is excluded only on days it is overbought and
becomes eligible again the moment it cools off — substitution and swap-back fall
out of the daily recompute automatically.

No look-ahead: the roster held on day ``t`` is chosen from the slow signal,
SMA-overbought flags, and analyze gate observed at ``t-1`` (decided at the prior
close, accruing on ``t``) — mirroring Phase 1's ``roster_lag=1``.

The analyze gate is exogenous and precomputed (``data/overbought_individual.csv``,
column ``analyze``); this module consumes it as-is and never recomputes it. Six
rows (2025-09-05..2025-09-12) of that file are imputed; see
``docs/individual_overbought_methodology.md``.

After the four subs are combined exactly as Phase 1 combines them (simple mean
of the active subs' daily returns, phased in by start date), the Phase 2 market
cash mask is applied **last and unchanged** via :mod:`src.overbought_filter`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, compute_simple_returns
from .metrics import summary
from .overbought_filter import apply_filter
from .signals import compute_risk_adj_return, compute_slow_signal
from .stack_backtest import (
    HEADLINE_START,
    N_SLOTS,
    SUB_STRATEGIES,
    SubStrategyConfig,
    SubStrategyResult,
    _resolve_start_idx,
)

#: Short / long simple-moving-average windows (trading days).
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50

#: Overbought multipliers applied to the short / long SMA.
SMA_SHORT_MULTIPLIER = 1.05
SMA_LONG_MULTIPLIER = 1.07

#: (sma20_pct, sma50_pct) pairs swept by :func:`run_threshold_sweep`. The first
#: pair is the canonical 5% / 7% screen; the rest loosen both legs together.
DEFAULT_THRESHOLD_GRID: tuple[tuple[float, float], ...] = (
    (0.05, 0.07),
    (0.06, 0.08),
    (0.07, 0.09),
    (0.08, 0.10),
    (0.10, 0.12),
)


def load_analyze_gate(path: Path | str | None = None) -> pd.Series:
    """Load the exogenous analyze gate from the individual-overbought file.

    Parameters
    ----------
    path : Path or str, optional
        Path to ``overbought_individual.csv``. Defaults to
        ``data/overbought_individual.csv``.

    Returns
    -------
    pandas.Series
        The ``analyze`` column (``0``/``1``) indexed by date, sorted ascending.
        ``1`` means the individual-ETF screen is active that day; ``0`` means it
        is off.
    """
    if path is None:
        path = DATA_DIR / "overbought_individual.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()["analyze"].astype(int)


def compute_overbought_flags(
    closes: pd.DataFrame,
    sma20_pct: float | None = None,
    sma50_pct: float | None = None,
) -> pd.DataFrame:
    """Flag each ETF on each day as individually overbought.

    An ETF is overbought when its close is at or above ``(1 + sma20_pct)`` x its
    ``SMA_SHORT_WINDOW``-day SMA **or** at or above ``(1 + sma50_pct)`` x its
    ``SMA_LONG_WINDOW``-day SMA. The comparison is inclusive (``>=``) on both
    legs.

    Parameters
    ----------
    closes : pandas.DataFrame
        Close prices indexed by date, one column per ticker.
    sma20_pct : float, optional
        Fractional stretch above the 20-day SMA that counts as overbought. The
        default ``None`` uses the canonical 5% threshold (the literal
        ``SMA_SHORT_MULTIPLIER`` multiplier), keeping the default path
        bit-identical to the frozen Phase 2b run. Any explicit value uses the
        multiplier ``1 + sma20_pct`` instead (e.g. ``0.06`` for 6%).
    sma50_pct : float, optional
        Fractional stretch above the 50-day SMA. ``None`` uses the canonical 7%
        threshold (``SMA_LONG_MULTIPLIER``); an explicit value uses
        ``1 + sma50_pct``.

    Returns
    -------
    pandas.DataFrame of bool
        ``True`` where the ETF is overbought, same shape as ``closes``.

    Notes
    -----
    Days without a full 20- or 50-day window produce NaN SMAs; comparisons
    against NaN evaluate to ``False``, so such ETFs are simply not flagged. No
    ``fillna`` or special-casing is applied.
    """
    short_mult = SMA_SHORT_MULTIPLIER if sma20_pct is None else 1.0 + sma20_pct
    long_mult = SMA_LONG_MULTIPLIER if sma50_pct is None else 1.0 + sma50_pct
    sma_short = closes.rolling(window=SMA_SHORT_WINDOW).mean()
    sma_long = closes.rolling(window=SMA_LONG_WINDOW).mean()
    short_hot = closes >= short_mult * sma_short
    long_hot = closes >= long_mult * sma_long
    return short_hot | long_hot


def _cooldown_bar(blocked: pd.DataFrame, cooldown_days: int) -> pd.DataFrame:
    """Bar each name for the ``cooldown_days`` trading days after a screen hit.

    A screen hit on decision day ``d`` (``blocked[d]`` is ``True``) bars the name
    on the following ``cooldown_days`` decision days ``d+1 .. d+cooldown_days``.
    The bar is the union of those windows over all hits, so a fresh hit while a
    name is still barred restarts the count and overlapping bars merge:
    ``barred[d'] = OR(blocked[d'-1], .., blocked[d'-cooldown_days])``.

    Parameters
    ----------
    blocked : pandas.DataFrame of bool
        Same-day screen hits per ETF (``overbought and analyze == 1``).
    cooldown_days : int
        Number of trading days a name stays barred after a hit (``> 0``).

    Returns
    -------
    pandas.DataFrame of bool
        ``True`` where a name is barred by the cooldown, aligned to ``blocked``.

    Notes
    -----
    Because the offsets are strictly positive (``1 .. cooldown_days``),
    ``barred[d']`` depends only on hits strictly before ``d'`` — preserving the
    no-look-ahead guarantee. This is the same shift-OR shape as
    :func:`src.overbought_filter.build_cash_mask`, with offsets ``1..N`` instead
    of the market filter's ``3..6``.
    """
    barred = pd.DataFrame(False, index=blocked.index, columns=blocked.columns)
    for k in range(1, cooldown_days + 1):
        barred |= blocked.shift(k, fill_value=False)
    return barred


def compute_eligibility(
    slow_signal: pd.DataFrame,
    overbought: pd.DataFrame,
    analyze_gate: pd.Series,
    screen: bool = True,
    cooldown_days: int = 0,
) -> pd.DataFrame:
    """Compute the per-day, per-ETF eligibility mask.

    An ETF is eligible on a day when its slow signal is strictly positive and,
    when the screen is active (``screen`` is ``True`` *and* ``analyze == 1`` that
    day), it is not individually overbought. With ``cooldown_days > 0`` a name
    also stays barred for that many trading days after each overbought flag.

    Parameters
    ----------
    slow_signal : pandas.DataFrame
        Slow signal per ETF (dates x tickers) for one sub-strategy's lookback.
    overbought : pandas.DataFrame
        Individual-overbought flags per ETF (dates x tickers).
    analyze_gate : pandas.Series
        The ``analyze`` gate (``0``/``1``) indexed by date.
    screen : bool, optional
        Whether to apply the individual-overbought screen at all (default
        ``True``). When ``False`` the eligibility reduces to ``slow_signal > 0``
        — the daily-rebalanced, screen-off diagnostic — and ``cooldown_days`` has
        no effect (there are no screen hits to start a cooldown).
    cooldown_days : int, optional
        Trading days a name remains barred after a screen hit (default ``0`` —
        the no-timer behavior in which a name is eligible again the moment it
        stops being overbought). Screen hits are gated by ``analyze``, so no new
        cooldown starts on ``analyze == 0`` days, though an existing bar keeps
        counting through them.

    Returns
    -------
    pandas.DataFrame of bool
        ``True`` where the ETF is eligible, aligned to ``slow_signal``.

    Notes
    -----
    Trading days absent from ``analyze_gate`` are treated as ``analyze == 0``
    (screen off) — the neutral default that never excludes a name on the basis
    of a missing gate value.
    """
    eligible = slow_signal > 0
    if screen:
        screen_active = analyze_gate.reindex(
            slow_signal.index, fill_value=0
        ).eq(1)
        blocked = overbought.mul(screen_active.astype(int), axis=0).astype(bool)
        exclusion = blocked
        if cooldown_days > 0:
            exclusion = exclusion | _cooldown_bar(blocked, cooldown_days)
        eligible = eligible & ~exclusion
    return eligible


def select_daily_roster(
    slow_signal: pd.DataFrame,
    eligible: pd.DataFrame,
    start_idx: int,
    n_slots: int = N_SLOTS,
) -> pd.DataFrame:
    """Pick each day's top-``n_slots`` eligible names by descending slow signal.

    For every trading day at or after ``start_idx`` the eligible ETFs are ranked
    by slow signal (descending, ties broken by column order — matching Phase 1's
    stable sort) and the top ``n_slots`` are selected. Fewer than ``n_slots``
    eligible names leave the remaining slots empty (cash); non-positive names are
    never reached because eligibility already requires ``slow_signal > 0``.

    Parameters
    ----------
    slow_signal : pandas.DataFrame
        Slow signal per ETF (dates x tickers).
    eligible : pandas.DataFrame of bool
        Per-day eligibility mask, aligned to ``slow_signal``.
    start_idx : int
        First trading-day position at which the sub-strategy selects a roster.
    n_slots : int, optional
        Number of slots to fill (default :data:`N_SLOTS`).

    Returns
    -------
    pandas.DataFrame of bool
        ``True`` where the ETF is selected on that day's close, aligned to
        ``slow_signal``. All ``False`` before ``start_idx``.
    """
    selected = pd.DataFrame(
        False, index=slow_signal.index, columns=slow_signal.columns
    )
    cols = selected.columns
    n = len(slow_signal)
    for d in range(start_idx, n):
        mask = eligible.iloc[d]
        if not mask.any():
            continue
        scores = slow_signal.iloc[d][mask]
        ranked = scores.sort_values(ascending=False, kind="stable")
        chosen = ranked.index[:n_slots]
        selected.iloc[d, cols.get_indexer(chosen)] = True
    return selected


def sub_returns_from_selection(
    simple_returns: pd.DataFrame,
    selected: pd.DataFrame,
    roster_lag: int = 1,
    n_slots: int = N_SLOTS,
) -> pd.Series:
    """Turn a daily selection matrix into a sub-strategy's daily return series.

    The roster selected at a day's close (``selected``) is shifted forward by
    ``roster_lag`` so it accrues on the following trading day(s). Each held name
    earns its simple return; empty slots earn zero. The sub return is the sum of
    the five slot returns divided by ``n_slots`` (the fixed divisor encodes the
    ``$20``-per-slot, ``$100``-capital, non-compounding convention).

    Parameters
    ----------
    simple_returns : pandas.DataFrame
        Daily simple returns per ETF (dates x tickers).
    selected : pandas.DataFrame of bool
        Per-day selection matrix from :func:`select_daily_roster`.
    roster_lag : int, optional
        Days between selecting a roster and its accrual (default ``1`` — the
        no-look-ahead lag mirroring Phase 1).
    n_slots : int, optional
        Slot count / fixed return divisor (default :data:`N_SLOTS`).

    Returns
    -------
    pandas.Series
        Daily sub-strategy returns, aligned to ``simple_returns`` and zero
        before the roster first accrues.
    """
    held = selected.shift(roster_lag, fill_value=False)
    return (simple_returns * held).sum(axis=1).div(n_slots)


def run_sub_strategy_2b(
    config: SubStrategyConfig,
    simple_returns: pd.DataFrame,
    risk_adj_return: pd.DataFrame,
    overbought: pd.DataFrame,
    analyze_gate: pd.Series,
    screen: bool = True,
    roster_lag: int = 1,
    cooldown_days: int = 0,
) -> SubStrategyResult:
    """Run one Phase 2b sub-strategy (daily-rebalanced, screened selection).

    Parameters
    ----------
    config : SubStrategyConfig
        Sub-strategy parameters (start date, lookback).
    simple_returns : pandas.DataFrame
        Daily simple returns per ETF.
    risk_adj_return : pandas.DataFrame
        Daily risk-adjusted return signal per ETF.
    overbought : pandas.DataFrame of bool
        Individual-overbought flags per ETF.
    analyze_gate : pandas.Series
        The ``analyze`` gate (``0``/``1``) indexed by date.
    screen : bool, optional
        Apply the individual-overbought screen (default ``True``).
    roster_lag : int, optional
        No-look-ahead roster lag (default ``1``).
    cooldown_days : int, optional
        Trading days a name stays barred after a screen hit (default ``0`` — no
        timer). See :func:`compute_eligibility`.

    Returns
    -------
    SubStrategyResult
        Daily returns over the full calendar (zero before the start) and the
        resolved start position.
    """
    dates = simple_returns.index
    start_idx = _resolve_start_idx(dates, config.start)

    slow_signal = compute_slow_signal(risk_adj_return, config.lookback)
    eligible = compute_eligibility(
        slow_signal,
        overbought,
        analyze_gate,
        screen=screen,
        cooldown_days=cooldown_days,
    )
    selected = select_daily_roster(slow_signal, eligible, start_idx)
    returns = sub_returns_from_selection(
        simple_returns, selected, roster_lag=roster_lag
    ).fillna(0.0)

    return SubStrategyResult(
        name=config.name, start_idx=start_idx, returns=returns
    )


def run_individual_overbought(
    closes: pd.DataFrame,
    configs: tuple[SubStrategyConfig, ...] = SUB_STRATEGIES,
    screen: bool = True,
    roster_lag: int = 1,
    cooldown_days: int = 0,
    sma20_pct: float | None = None,
    sma50_pct: float | None = None,
) -> dict[str, object]:
    """Run the daily-rebalanced Phase 2b portfolio (before the market mask).

    The four sub-strategies are combined exactly as Phase 1 combines them: the
    simple mean of the active subs' daily returns, with each sub phased in from
    its start date. The Phase 2 market cash mask is **not** applied here — the
    caller applies it last via :mod:`src.overbought_filter` so the unmasked
    series can also serve as a diagnostic.

    Parameters
    ----------
    closes : pandas.DataFrame
        Close prices indexed by date, one column per ticker.
    configs : tuple of SubStrategyConfig, optional
        Sub-strategy configurations (defaults to Phase 1's A, B, C, D).
    screen : bool, optional
        Apply the individual-overbought screen (default ``True``). Set ``False``
        for the daily-rebalanced, screen-off diagnostic.
    roster_lag : int, optional
        No-look-ahead roster lag (default ``1``).
    cooldown_days : int, optional
        Trading days a name stays barred after a screen hit (default ``0`` — no
        timer; the original Phase 2b behavior). See :func:`compute_eligibility`.
    sma20_pct, sma50_pct : float, optional
        Overbought-threshold overrides forwarded to
        :func:`compute_overbought_flags`. ``None`` (default) uses the canonical
        5% / 7% thresholds, keeping this run bit-identical to the frozen Phase 2b.

    Returns
    -------
    dict
        ``portfolio_returns`` (pandas.Series) — combined daily returns over the
        full calendar; ``sub_results`` (dict of name -> SubStrategyResult);
        ``headline_returns`` (pandas.Series) — portfolio returns from the
        headline start date onward. No market mask applied.
    """
    simple_returns = compute_simple_returns(closes)
    risk_adj_return = compute_risk_adj_return(closes)
    overbought = compute_overbought_flags(closes, sma20_pct, sma50_pct)
    analyze_gate = load_analyze_gate()

    sub_results = {
        cfg.name: run_sub_strategy_2b(
            cfg,
            simple_returns,
            risk_adj_return,
            overbought,
            analyze_gate,
            screen=screen,
            roster_lag=roster_lag,
            cooldown_days=cooldown_days,
        )
        for cfg in configs
    }

    dates = closes.index
    sub_returns = pd.DataFrame(
        {name: res.returns for name, res in sub_results.items()}, index=dates
    )
    active = pd.DataFrame(
        {
            name: (np.arange(len(dates)) >= res.start_idx).astype(float)
            for name, res in sub_results.items()
        },
        index=dates,
    )
    n_active = active.sum(axis=1)
    combined = (sub_returns * active).sum(axis=1) / n_active.replace(0.0, np.nan)
    portfolio_returns = combined.fillna(0.0)

    headline_returns = portfolio_returns.loc[HEADLINE_START:]

    return {
        "portfolio_returns": portfolio_returns,
        "sub_results": sub_results,
        "headline_returns": headline_returns,
    }


def run_threshold_sweep(
    closes: pd.DataFrame,
    cash_mask: pd.Series,
    grid: tuple[tuple[float, float], ...] = DEFAULT_THRESHOLD_GRID,
    cooldown_days: int = 0,
) -> pd.DataFrame:
    """Sweep the individual-overbought thresholds and tabulate headline metrics.

    For each ``(sma20_pct, sma50_pct)`` pair the full Phase 2b backtest is run
    (all four sub-strategies, screen on, the given cooldown) and the Phase 2
    market cash mask is applied last — reusing :func:`run_individual_overbought`,
    :func:`src.overbought_filter.apply_filter`, and :func:`src.metrics.summary`
    rather than re-implementing any backtest logic. This is a sensitivity
    analysis: the full grid is reported and no threshold is adopted.

    Parameters
    ----------
    closes : pandas.DataFrame
        Close prices indexed by date, one column per ticker.
    cash_mask : pandas.Series of bool
        The Phase 2 market cash mask, applied last to each variant.
    grid : tuple of (float, float), optional
        ``(sma20_pct, sma50_pct)`` pairs to evaluate (default
        :data:`DEFAULT_THRESHOLD_GRID`). The first pair should be the canonical
        ``(0.05, 0.07)`` so the sweep includes the shipped Phase 2b screen.
    cooldown_days : int, optional
        Cooldown forwarded to each run (default ``0`` — no timer, isolating the
        threshold effect).

    Returns
    -------
    pandas.DataFrame
        One row per pair with columns ``sma20_pct``, ``sma50_pct``,
        ``total_return``, ``sharpe``, ``max_dd`` (headline window).
    """
    rows = []
    for sma20_pct, sma50_pct in grid:
        result = run_individual_overbought(
            closes,
            screen=True,
            cooldown_days=cooldown_days,
            sma20_pct=sma20_pct,
            sma50_pct=sma50_pct,
        )
        filtered = apply_filter(result["headline_returns"], cash_mask)
        metrics = summary(filtered)
        rows.append(
            {
                "sma20_pct": sma20_pct,
                "sma50_pct": sma50_pct,
                "total_return": metrics["total_return"],
                "sharpe": metrics["sharpe_ratio"],
                "max_dd": metrics["max_drawdown"],
            }
        )
    return pd.DataFrame(rows, columns=[
        "sma20_pct", "sma50_pct", "total_return", "sharpe", "max_dd"
    ])
