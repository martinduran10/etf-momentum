# Project briefing — ETF Momentum (Phase 1: Stack Portfolio)

## Goal

Faithfully reproduce an Excel "Stack Portfolio" tactical ETF allocation
backtest in Python. Phase 1 is **reproduction only** — no regime detection,
market filters, walk-forward, factor regression, transaction costs, or other
improvements. Those are future phases. Match the Excel mechanics literally.

## Universe

The 43 ETFs in `data/universe.csv` (global equity, US sectors/styles, single
countries, regions, REITs, US fixed income). Daily closes in
`data/closes_wide.csv`, 2014-07-03 → 2026-05-22 (2,990 trading days). Splits
are already back-adjusted in the closes. **Never** import the precomputed
columns from `etf_panel.parquet`; recompute everything from raw closes.

## Signal formulas

Per ETF, per day `t`:

```
log_ret_t        = ln(close_t / close_{t-1})
vol_10_t         = rolling_std(log_ret, 10, ddof=1) * sqrt(252)   # annualized
risk_adj_return_t = log_ret_t / vol_10_t            # NaN if vol_10 is 0/undef
slow_signal_t    = rolling_mean(risk_adj_return, lookback)
```

## Mechanics summary

Four sub-strategies (A/B/C/D) run in parallel, identical except start date and
lookback:

| Sub | Start (resolved) | Lookback |
|-----|------------------|----------|
| A   | 2015-12-07       | 260      |
| B   | 2015-12-11       | 280      |
| C   | 2015-12-18       | 300      |
| D   | 2015-12-28*      | 320      |

*2015-12-25 is a holiday → next trading day.

Each sub: rebalances every 20 trading days from its start. At a rebalance it
ranks all 43 ETFs by `slow_signal` and takes the top 5 that are also strictly
positive (fewer if fewer positive; empty slots stay cash). $100 capital, five
$20 slots, **no compounding** (capital resets to $100 each rebalance, slot size
fixed). Between rebalances each slot toggles daily: invested (earns that day's
simple close-to-close return) when its `slow_signal > 0`, else cash (0). A
sub's daily return is the sum of slot returns / 5. The combined portfolio
return is the simple mean of the active subs' returns (phase-in as each sub
starts).

### Timing convention (matters for reproduction)

- **Roster selection** uses the slow signal at the rebalance close, but the
  roster begins accruing returns the **next** day (`roster_lag=1`) — honoring
  no-look-ahead on position-taking.
- **Daily in/out toggle** uses the **same-day** slow signal (`signal_lag=0`) —
  the literal Excel step-4 mechanic.

This combination reproduces the Excel headline within tolerance; other
combinations do not.

## Metrics (arithmetic, no compounding)

`total_return = sum(daily)`, `annualized_return = mean(daily)*252`,
`annualized_volatility = std(daily, ddof=1)*sqrt(252)`,
`sharpe = ann_return / ann_vol` (rf=0), `max_drawdown` from the cumulative-sum
curve. Headline window: 2015-12-07 → end of data.

### Validation targets (CI gate, ±3% relative)

total 108.58% · ann 10.82% · vol 16.16% · Sharpe 0.67 · max DD −28.49%.

## Layout

- `src/data.py` — load closes, log/simple returns, `vol_10`.
- `src/signals.py` — `compute_risk_adj_return`, `compute_slow_signal`.
- `src/stack_backtest.py` — sub-strategies, rebalance, dynamic re-entry, combine.
- `src/metrics.py` — performance metrics.
- `tests/` — data, signals, mechanics, and `test_validation.py` (the CI gate).
- `scripts/run_stack_backtest.py` — end-to-end run → figures + tables.

## Conventions

Python 3.11+, type hints, NumPy-style docstrings. No print statements in
library code (scripts may print). No external data downloads — only `data/`.
Do not reference the `legacy-v1` branch. Run `pytest tests/ -v` before
committing.
