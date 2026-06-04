# Project briefing — ETF Momentum (Phases 1–2: Stack Portfolio + Overbought Filter)

## Goal

Faithfully reproduce an Excel "Stack Portfolio" tactical ETF allocation
backtest in Python, then layer the strategy's later phases on top as additive
overlays. **Phase 1 (Stack Portfolio) is reproduction only** — no regime
detection, walk-forward, factor regression, transaction costs, or other
improvements baked into it; match the Excel mechanics literally and keep it
frozen. The **overbought market filter is Phase 2** (built — see below),
applied as an overlay that never modifies Phase 1. The divergence filter
(Phase III) and risk-on/risk-off variable (Phase IV) remain future work.
Faithful reproduction always takes priority over "improvements."

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

## Phase 2 — Overbought market filter (additive overlay)

Phase 2 sits **on top of** Phase 1's combined daily-return series; the Phase 1
engine, sub-strategies, and outputs are left byte-for-byte unchanged (a
frozen-snapshot test enforces this).

**Signal.** An exogenous, precomputed binary series in
`data/overbought_signal.csv` (`date, market_ok`; `1` = OK to trade, `0` =
overbought). Python consumes it as-is and does **not** recompute it. How the
values were derived is documented in `docs/overbought_methodology.md`.

**Cash-window rule (custom variant, not the deck's T+1).** For each trading day
`T` where `market_ok == 0`, force cash on the four trading days `T+3, T+4, T+5,
T+6`. Offsets are counted in the strategy's trading-day calendar
(weekends/holidays skipped). Overlapping windows merge. The 3-day minimum lag
guarantees no look-ahead (day `t`'s cash decision uses only `market_ok` at
`t-3` and earlier). A lookback offset predating the filter data, or a strategy
trading day with no matching date in the filter CSV, is treated as
`market_ok == 1` (OK).

**Application.** `filtered_return[t] = 0` on cash days, else `phase1_return[t]`.
Phase 1 keeps running underneath during cash windows — rebalances still happen,
the in/out toggle and same-day-close re-entry keep operating; Phase 2 only masks
the output.

**No validation target.** Phase 2 runs in empirical mode (the custom T+3 variant
vs the deck's T+1), so there is no CI metric gate for it. Results closely track
the deck's published Phase II (134.94% / Sharpe 1.01):

| Metric | Unfiltered (Phase 1) | Filtered (Phase 2) |
|--------|----------------------|--------------------|
| Total return | 110.99% | 136.18% |
| Annualized return | 10.63% | 13.05% |
| Annualized volatility | 16.21% | 13.40% |
| Sharpe ratio | 0.66 | 0.97 |
| Max drawdown | −28.54% | −23.90% |
| % of days in cash | — | 36.96% |

## Layout

- `src/data.py` — load closes, log/simple returns, `vol_10`.
- `src/signals.py` — `compute_risk_adj_return`, `compute_slow_signal`.
- `src/stack_backtest.py` — Phase 1 engine; sub-strategies, rebalance, dynamic
  re-entry, combine. **Frozen — do not modify.**
- `src/metrics.py` — performance metrics.
- `src/benchmarks.py` — SPY buy-and-hold comparison.
- `src/overbought_filter.py` — Phase 2 overlay (load signal, build cash mask,
  apply filter).
- `data/overbought_signal.csv` — Phase 2 input (`date, market_ok`).
- `docs/overbought_methodology.md` — how `market_ok` was derived.
- `tests/` — data, signals, mechanics, overbought-filter tests, a
  frozen-snapshot test pinning Phase 1 byte-for-byte, and `test_validation.py`
  (the Phase 1 CI gate).
- `scripts/run_stack_backtest.py` — end-to-end run → figures + tables (incl. the
  Phase 2 filtered-vs-unfiltered outputs).

## Conventions

Python 3.11+, type hints, NumPy-style docstrings. No print statements in
library code (scripts may print). No external data downloads — only `data/`.
Each phase is a **purely additive overlay** — never modify a prior phase's code
or outputs (a frozen-snapshot test guards Phase 1). Do not reference the
`legacy-v1` branch. Run `pytest tests/ -v` before committing.
