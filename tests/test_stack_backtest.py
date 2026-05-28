"""Tests for the Stack Portfolio engine mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import compute_simple_returns
from src.signals import compute_risk_adj_return
from src.stack_backtest import (
    N_SLOTS,
    REBALANCE_EVERY,
    SUB_STRATEGIES,
    SubStrategyConfig,
    _resolve_start_idx,
    run_stack_portfolio,
    run_sub_strategy,
)


def _make_closes(n: int = 700, n_etfs: int = 43, seed: int = 0) -> pd.DataFrame:
    """Build a deterministic synthetic close-price panel."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-07-03", periods=n, freq="B")
    tickers = [f"e{i:02d}" for i in range(n_etfs)]
    rets = rng.normal(0.0003, 0.01, size=(n, n_etfs))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_start_date_resolves_holiday_to_next_trading_day() -> None:
    dates = pd.DatetimeIndex(
        pd.to_datetime(["2015-12-23", "2015-12-24", "2015-12-28", "2015-12-29"])
    )
    # 2015-12-25 is a market holiday -> next trading day is 2015-12-28 (idx 2).
    assert _resolve_start_idx(dates, "2015-12-25") == 2


def test_roster_is_top5_strictly_positive() -> None:
    closes = _make_closes()
    sret = compute_simple_returns(closes)
    rar = compute_risk_adj_return(closes)
    cfg = SubStrategyConfig("T", str(closes.index[400].date()), 260)
    from src.signals import compute_slow_signal

    res = run_sub_strategy(cfg, sret, rar)
    slow = compute_slow_signal(rar, cfg.lookback)
    for date, roster in res.rosters.items():
        scores = slow.loc[date]
        assert len(roster) <= N_SLOTS
        # Every chosen name has a strictly positive slow signal.
        assert all(scores[t] > 0 for t in roster)
        # No omitted, positive, higher-scoring name was skipped.
        if roster:
            cutoff = scores[roster].min()
            higher = scores[(scores > cutoff)].dropna()
            assert set(higher.index).issubset(set(roster))


def test_no_more_than_five_positive_means_fewer_held() -> None:
    # Construct a panel where at the rebalance only 2 names are positive.
    closes = _make_closes(seed=3)
    sret = compute_simple_returns(closes)
    rar = compute_risk_adj_return(closes)
    cfg = SubStrategyConfig("T", str(closes.index[400].date()), 260)
    from src.signals import compute_slow_signal

    slow = compute_slow_signal(rar, cfg.lookback)
    res = run_sub_strategy(cfg, sret, rar)
    for date, roster in res.rosters.items():
        n_positive = int((slow.loc[date] > 0).sum())
        assert len(roster) == min(N_SLOTS, n_positive)


def test_no_compounding_daily_return_is_mean_of_slots() -> None:
    """A sub-strategy's daily return equals (sum of slot returns)/5."""
    closes = _make_closes()
    sret = compute_simple_returns(closes)
    rar = compute_risk_adj_return(closes)
    from src.signals import compute_slow_signal

    cfg = SubStrategyConfig("T", str(closes.index[400].date()), 260)
    res = run_sub_strategy(cfg, sret, rar, signal_lag=0, roster_lag=1)
    slow = compute_slow_signal(rar, cfg.lookback)

    # Reconstruct one in-window day by hand.
    r_idx = res.start_idx
    roster = res.rosters[closes.index[r_idx]]
    day = r_idx + 5  # inside the first holding window
    manual = 0.0
    for t in roster:
        if slow[t].iloc[day] > 0:
            manual += sret[t].iloc[day]
    manual /= N_SLOTS
    assert np.isclose(res.returns.iloc[day], manual)


def test_cash_when_signal_not_positive() -> None:
    """A slot whose slow signal is <= 0 contributes zero that day."""
    closes = _make_closes(seed=7)
    sret = compute_simple_returns(closes)
    rar = compute_risk_adj_return(closes)
    from src.signals import compute_slow_signal

    cfg = SubStrategyConfig("T", str(closes.index[400].date()), 260)
    res = run_sub_strategy(cfg, sret, rar)
    slow = compute_slow_signal(rar, cfg.lookback)
    r_idx = res.start_idx
    roster = res.rosters[closes.index[r_idx]]
    # Find a day where at least one roster name has signal <= 0.
    for day in range(r_idx + 1, r_idx + REBALANCE_EVERY):
        contrib = {t: sret[t].iloc[day] for t in roster if slow[t].iloc[day] > 0}
        expected = sum(contrib.values()) / N_SLOTS
        assert np.isclose(res.returns.iloc[day], expected)


def test_no_lookahead_positions_accrue_day_after_selection() -> None:
    """With roster_lag=1, the rebalance day itself earns no new-roster return."""
    closes = _make_closes()
    sret = compute_simple_returns(closes)
    rar = compute_risk_adj_return(closes)
    cfg = SubStrategyConfig("T", str(closes.index[400].date()), 260)
    res = run_sub_strategy(cfg, sret, rar, roster_lag=1)
    # The very first rebalance day has no prior roster -> zero return.
    assert res.returns.iloc[res.start_idx] == 0.0


def test_returns_zero_before_start() -> None:
    closes = _make_closes()
    sret = compute_simple_returns(closes)
    rar = compute_risk_adj_return(closes)
    cfg = SubStrategyConfig("T", str(closes.index[400].date()), 260)
    res = run_sub_strategy(cfg, sret, rar)
    assert (res.returns.iloc[: res.start_idx] == 0.0).all()


def test_rebalance_cadence_is_20_days() -> None:
    closes = _make_closes()
    sret = compute_simple_returns(closes)
    rar = compute_risk_adj_return(closes)
    cfg = SubStrategyConfig("T", str(closes.index[400].date()), 260)
    res = run_sub_strategy(cfg, sret, rar)
    idxs = [closes.index.get_loc(d) for d in res.rosters]
    diffs = np.diff(idxs)
    assert (diffs == REBALANCE_EVERY).all()


def test_phase_in_combination_weights() -> None:
    """Before all four subs start, the portfolio averages only active subs."""
    closes = _make_closes(n=700)
    res = run_stack_portfolio(closes)
    subs = res["sub_results"]
    port = res["portfolio_returns"]
    # On a day where only A and B have started, port == mean(A, B).
    start_b = subs["B"].start_idx
    start_c = subs["C"].start_idx
    mid = (start_b + start_c) // 2
    day = closes.index[mid]
    expected = np.mean([subs[s].returns.loc[day] for s in ("A", "B")])
    assert np.isclose(port.loc[day], expected)


def test_default_subs_have_expected_lookbacks() -> None:
    lookbacks = {c.name: c.lookback for c in SUB_STRATEGIES}
    assert lookbacks == {"A": 260, "B": 280, "C": 300, "D": 320}
