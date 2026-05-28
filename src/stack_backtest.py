"""Stack Portfolio backtest engine.

Four mechanically identical sub-strategies (A, B, C, D) run in parallel,
differing only by start date and lookback. Each holds an "eligible roster" of
up to five ETFs chosen every 20 trading days; between rebalances each $20 slot
toggles in and out of cash daily based on the sign of its slow signal. Capital
resets to $100 each rebalance and never compounds, so a sub-strategy's daily
return is the simple arithmetic mean of its five slot returns. The combined
portfolio return is the mean of the active sub-strategies' returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import compute_simple_returns
from .signals import compute_risk_adj_return, compute_slow_signal

#: Number of $20 slots per sub-strategy (total capital $100, no compounding).
N_SLOTS = 5

#: Trading days between rebalances.
REBALANCE_EVERY = 20


@dataclass(frozen=True)
class SubStrategyConfig:
    """Configuration for a single sub-strategy.

    Attributes
    ----------
    name : str
        Label (``"A"``..``"D"``).
    start : str
        Nominal start date (``YYYY-MM-DD``). Resolved to the first trading day
        on or after this date.
    lookback : int
        Slow-signal rolling-mean window in trading days.
    """

    name: str
    start: str
    lookback: int


#: The four sub-strategies of the Stack Portfolio.
SUB_STRATEGIES: tuple[SubStrategyConfig, ...] = (
    SubStrategyConfig("A", "2015-12-07", 260),
    SubStrategyConfig("B", "2015-12-11", 280),
    SubStrategyConfig("C", "2015-12-18", 300),
    SubStrategyConfig("D", "2015-12-25", 320),  # holiday -> next trading day
)

#: Headline measurement window start.
HEADLINE_START = "2015-12-07"


@dataclass
class SubStrategyResult:
    """Result of running one sub-strategy.

    Attributes
    ----------
    name : str
        Sub-strategy label.
    start_idx : int
        Integer position of the (resolved) start date in the trading calendar.
    returns : pandas.Series
        Daily sub-strategy returns, indexed by the full trading calendar.
        Zero before the start date.
    rosters : dict
        Mapping of each rebalance date to its selected list of tickers.
    """

    name: str
    start_idx: int
    returns: pd.Series
    rosters: dict[pd.Timestamp, list[str]] = field(default_factory=dict)


def _resolve_start_idx(dates: pd.DatetimeIndex, nominal: str) -> int:
    """Return the index of the first trading day on or after ``nominal``."""
    target = pd.Timestamp(nominal)
    pos = dates.searchsorted(target, side="left")
    if pos >= len(dates):
        raise ValueError(f"start date {nominal} is past the data range")
    return int(pos)


def run_sub_strategy(
    config: SubStrategyConfig,
    simple_returns: pd.DataFrame,
    risk_adj_return: pd.DataFrame,
    signal_lag: int = 0,
    roster_lag: int = 1,
) -> SubStrategyResult:
    """Run a single sub-strategy over the full calendar.

    Parameters
    ----------
    config : SubStrategyConfig
        Sub-strategy parameters.
    simple_returns : pandas.DataFrame
        Daily simple returns per ETF (dates x tickers).
    risk_adj_return : pandas.DataFrame
        Daily risk-adjusted return signal per ETF.
    signal_lag : int, optional
        Days by which the slow signal is lagged when gating daily slot
        positions. ``0`` (default) means a day's own slow signal gates that
        day's return — the literal Excel same-day toggle of step 4.
    roster_lag : int, optional
        Days between selecting a roster (at the rebalance close) and that
        roster beginning to accrue returns. ``1`` (default) honors the
        no-look-ahead rule: positions taken at day *t*'s close start earning on
        day *t+1*.

    Returns
    -------
    SubStrategyResult
        Daily returns and the per-rebalance rosters.

    Notes
    -----
    Roster selection at a rebalance date uses the slow signal observed on that
    date. The eligible roster is held until the next rebalance, beginning to
    accrue ``roster_lag`` days later. Within the holding window each slot earns
    the ETF's simple return on days its (optionally lagged) slow signal is
    strictly positive, and zero otherwise; unfilled slots (fewer than five
    positive names at the rebalance) are always cash. The sub-strategy's daily
    return is the sum of the slot returns divided by ``N_SLOTS``.
    """
    dates = simple_returns.index
    n = len(dates)
    start_idx = _resolve_start_idx(dates, config.start)

    slow_signal = compute_slow_signal(risk_adj_return, config.lookback)
    gate_signal = slow_signal.shift(signal_lag) if signal_lag else slow_signal

    rebalance_idxs = list(range(start_idx, n, REBALANCE_EVERY))

    returns = pd.Series(0.0, index=dates)
    rosters: dict[pd.Timestamp, list[str]] = {}

    for k, r_idx in enumerate(rebalance_idxs):
        # Roster: top-5 ETFs by slow signal on the rebalance date, kept only if
        # strictly positive. Ranking is deterministic: ties broken by ticker.
        scores = slow_signal.iloc[r_idx].dropna()
        scores = scores[scores > 0]
        ranked = scores.sort_values(ascending=False, kind="stable")
        roster = list(ranked.index[:N_SLOTS])
        rosters[dates[r_idx]] = roster

        next_idx = (
            rebalance_idxs[k + 1] if k + 1 < len(rebalance_idxs) else n
        )
        if not roster:
            continue  # all slots in cash this window -> zero returns

        # Returns accrue from roster_lag days after the rebalance until the next
        # roster takes over (also lagged), keeping the windows contiguous.
        beg = min(r_idx + roster_lag, n)
        end = min(next_idx + roster_lag, n)
        if beg >= end:
            continue

        window = slice(beg, end)
        window_ret = simple_returns.iloc[window][roster]
        window_gate = gate_signal.iloc[window][roster] > 0
        # Slot return = ETF return when in position, else 0; mean over N_SLOTS
        # slots (missing names are permanent cash, hence the fixed divisor).
        slot_pnl = (window_ret * window_gate).sum(axis=1) / N_SLOTS
        returns.iloc[window] = slot_pnl.values

    return SubStrategyResult(
        name=config.name,
        start_idx=start_idx,
        returns=returns,
        rosters=rosters,
    )


def run_stack_portfolio(
    closes: pd.DataFrame,
    configs: tuple[SubStrategyConfig, ...] = SUB_STRATEGIES,
    signal_lag: int = 0,
    roster_lag: int = 1,
) -> dict[str, object]:
    """Run the full Stack Portfolio and combine the sub-strategies.

    Parameters
    ----------
    closes : pandas.DataFrame
        Close prices indexed by date, one column per ticker.
    configs : tuple of SubStrategyConfig, optional
        Sub-strategy configurations (defaults to A, B, C, D).
    signal_lag : int, optional
        Slow-signal lag applied when gating daily slot positions (default 0).
    roster_lag : int, optional
        Days between roster selection and the start of return accrual
        (default 1; honors the no-look-ahead rule).

    Returns
    -------
    dict
        ``portfolio_returns`` (pandas.Series) — combined daily returns over the
        full calendar; ``sub_results`` (dict of name -> SubStrategyResult);
        ``headline_returns`` (pandas.Series) — portfolio returns from the
        headline start date onward.
    """
    simple_returns = compute_simple_returns(closes)
    risk_adj_return = compute_risk_adj_return(closes)

    sub_results = {
        cfg.name: run_sub_strategy(
            cfg, simple_returns, risk_adj_return, signal_lag, roster_lag
        )
        for cfg in configs
    }

    dates = closes.index
    # Active-count per day: number of sub-strategies started on or before t.
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
