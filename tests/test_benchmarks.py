"""Tests for the SPY buy-and-hold benchmark."""

from __future__ import annotations

import pandas as pd

from src.benchmarks import spy_buy_and_hold_returns
from src.data import load_closes
from src.stack_backtest import HEADLINE_START, run_stack_portfolio


def test_spy_aligns_with_strategy_length_and_start() -> None:
    closes = load_closes()
    strategy_returns = run_stack_portfolio(closes)["headline_returns"]
    spy_returns = spy_buy_and_hold_returns(closes)

    assert len(spy_returns) == len(strategy_returns)
    assert spy_returns.index[0] == pd.Timestamp(HEADLINE_START)
    assert spy_returns.index.equals(strategy_returns.index)
