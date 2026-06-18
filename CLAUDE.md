# Project briefing — ETF Momentum (Phases 1–3: Stack Portfolio + Overbought & Divergence Filters)

## Goal

Faithfully reproduce an Excel "Stack Portfolio" tactical ETF allocation
backtest in Python, then layer the strategy's later phases on top as additive
overlays. **Phase 1 (Stack Portfolio) is reproduction only** — no regime
detection, walk-forward, factor regression, transaction costs, or other
improvements baked into it; match the Excel mechanics literally and keep it
frozen. The **overbought market filter is Phase 2** (built — see below),
applied as an overlay that never modifies Phase 1. The **individual-ETF
overbought filter is Phase 2b** (built — see below), a daily-rebalanced overlay
that reuses Phase 1/2 functions and modifies neither. The **divergence filter is
Phase 3** (built — see below), a market-level cash overlay stacked on Phase 2's
output. The risk-on/risk-off variable (Phase IV) remains future work.
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

This combination reproduces the Excel headline closely; other combinations do
not.

## Metrics (arithmetic, no compounding)

`total_return = sum(daily)`, `annualized_return = mean(daily)*252`,
`annualized_volatility = std(daily, ddof=1)*sqrt(252)`,
`sharpe = ann_return / ann_vol` (rf=0), `max_drawdown` from the cumulative-sum
curve. Headline window: 2015-12-07 → end of data.

### Reporting & regression

Phases are reported **empirically** — there is no target CI gate. The Excel
numbers survive only as a descriptive reproduction note in `RESULTS.md`/
`README.md`. Prior phases are pinned by **frozen-snapshot tests**:
`test_phase1_frozen.py` (Phase 1 headline metrics) and `test_validation.py`
(Phase 1 and Phase 2 daily-return series, byte-for-byte). Any overlay that
perturbs an earlier phase fails these.

## Phase 2 — Overbought market filter (additive overlay)

Phase 2 sits **on top of** Phase 1's combined daily-return series; the Phase 1
engine, sub-strategies, and outputs are left byte-for-byte unchanged (a
frozen-snapshot test enforces this).

**Signal.** An exogenous, precomputed binary series in
`data/overbought_signal.csv` (`date, market_ok`; `1` = OK to trade, `0` =
overbought). Python consumes it as-is and does **not** recompute it. How the
values were derived is documented in `docs/overbought_methodology.md`.

**Cash-window rule (T+3..T+6).** For each trading day
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

**No validation target.** Phase 2 runs in empirical mode, so there is no CI
metric gate for it:

| Metric | Unfiltered (Phase 1) | Filtered (Phase 2) |
|--------|----------------------|--------------------|
| Total return | 110.99% | 136.18% |
| Annualized return | 10.63% | 13.05% |
| Annualized volatility | 16.21% | 13.40% |
| Sharpe ratio | 0.66 | 0.97 |
| Max drawdown | −28.54% | −23.90% |
| % of days in cash | — | 36.96% |

## Phase 3 — Divergence filter (additive overlay)

Phase 3 stacks a market-level cash overlay on the main line (Phase 1 base returns
+ Phase 2 mask): cash whenever the Phase 2 mask **or** the divergence mask fires.
Phase 1/2 are consumed read-only (frozen-snapshot tests enforce this).

**Signal.** A divergence percentile in `data/divergence_percentile.csv` (`date,
divergence_pctl`; the top-5 vs bottom-5 momentum spread, in `[0, 1]`). Once an
exogenous CSV, it is now **reproduced in-repo from raw closes** by
`src/divergence_signal.py` (regenerate via `scripts/build_divergence_signal.py`) —
point-in-time with an **expanding percentile** and **no look-ahead** (each day
ranks only against history through that day). Guarded by
`tests/test_divergence_signal.py::test_no_look_ahead_truncation_invariance`
(truncating input history leaves earlier percentiles unchanged). The backtest
consumes the percentile from the CSV.

**State machine (T+1, no look-ahead).** Edge-triggered exit on an upward cross of
0.85 (`pctl[t-1] >= 0.85` and `pctl[t-2] < 0.85`); re-entry when `pctl[t-1] <
0.85` **or** the Phase 1 base curve (cumulative arithmetic return, percentage
points) has fallen >= 8 from its peak since the exit; re-arm needs a fresh cross.
Starts IN on 2015-12-07. Threshold 0.85 (exit `>=`, signal re-entry strict `<`),
reentry drop 8.0.

**No validation target** (empirical mode; tests pin mechanics, not numbers):

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| Total return | 110.99% | 136.18% | 167.85% |
| Annualized return | 10.63% | 13.05% | 16.08% |
| Annualized volatility | 16.21% | 13.40% | 12.25% |
| Sharpe ratio | 0.66 | 0.97 | 1.31 |
| Max drawdown | −28.54% | −23.90% | −18.26% |
| % of days in cash | — | 36.96% | 43.19% |

Cash-day split (share of all days): overbought-only 32.85%, divergence-only
6.24%, both 4.11%. Layered on Phase 2 it raises return and Sharpe at lower
volatility and a materially shallower drawdown (−23.90% → −18.26%).

## Phase 2b — Individual-ETF overbought filter (additive overlay)

Phase 2b is a second additive overlay. It reuses Phase 1/2 functions by
importing them (`signals`, `data`, `metrics`, `stack_backtest`,
`overbought_filter`) and modifies no existing module, test, or Phase 1/2
artifact. Unlike Phase 1's 20-day rebalance, **Phase 2b rebalances daily**.

**Per sub-strategy (A/B/C/D, same starts/lookbacks), per trading day `t`:**

- Rank all 43 ETFs by `slow_signal` (that sub's lookback). The ranking is
  recomputed **daily** — intentional.
- **Eligible** = `slow_signal > 0`, **and** (only when `analyze == 1` that day)
  **not** individually overbought.
- **Individual overbought** (per ETF, that day): `close >= 1.05 * SMA20` **or**
  `close >= 1.07 * SMA50`, where SMA20/SMA50 are **simple** moving averages of
  that ETF's own close over 20 and 50 trading days. Either leg true ⇒ excluded.
  An ETF without a full window is not flagged (NaN comparisons are `False`; no
  fillna, no special-case).
- **No timer by default** (`cooldown_days=0`): a name is excluded only on days it
  is overbought; eligible again the moment it is not (swap-back is automatic from
  the daily recompute). An optional `cooldown_days=N` additionally bars a flagged
  name for the next N trading days (a fresh flag resets the clock, overlapping
  bars merge, the bar persists across `analyze==0` days but no new bar starts
  there). `cooldown_days=0` is a strict no-op that reproduces the base Phase 2b.
- Hold the **top 5 eligible** names by descending `slow_signal`; walk down the
  ranking to fill 5 slots from eligible names only, remaining slots cash. Never
  reach into `slow_signal <= 0` names.
- $20/slot, $100, **no compounding**; sub daily return = sum of 5 slot returns
  / 5 (cash slots = 0).
- **No look-ahead**: the roster held on day `t` is chosen from `slow_signal`,
  SMA-overbought, and `analyze` as of `t-1` (decided at the prior close, accrues
  on `t`) — mirroring `roster_lag=1`.

**Signal.** The analyze gate is exogenous/precomputed in
`data/overbought_individual.csv` (columns `date, market_overbought, cancel,
overbought_net, analyze`); only `date` and `analyze` are used (`1` ⇒ screen
active). Consumed as-is, never recomputed. Six rows (2025-09-05 … 2025-09-12)
are imputed. Derivation in `docs/individual_overbought_methodology.md`.

**Combine & mask.** Subs combine exactly as Phase 1 (simple mean of active subs,
phased in by start date). The **Phase 2 market cash mask is applied last and
unchanged** (reuse `overbought_filter`): a Phase 2 cash day is cash regardless of
the 2b roster.

**Attribution / diagnostic series.** Two screen-off series isolate the drivers:
"daily-rebal + mask (screen off)" (daily top-5 on positive signal + the Phase 2
mask, no individual screen) and "daily-rebal (screen off)" (same, no mask). A
"Phase 2b + 5d cooldown" variant (`cooldown_days=5`) is also reported.

**No validation target** (empirical mode; tests pin mechanics, not numbers). Over
the headline window the highest Sharpe is Phase 1+2 (0.97); neither the
individual screen nor the 5-day cooldown raises risk-adjusted return here (Phase
2b 0.86 < daily-rebal+mask 0.91; the cooldown is lower than no-timer Phase 2b on
every metric):

| Metric | Phase 1 | Phase 1+2 | Phase 2b | Phase 2b + 5d cooldown | Daily-rebal + mask (screen off) | Daily-rebal (screen off) |
|--------|---------|-----------|----------|------------------------|---------------------------------|--------------------------|
| Total return | 110.99% | 136.18% | 124.30% | 120.39% | 126.80% | 108.61% |
| Annualized return | 10.63% | 13.05% | 11.91% | 11.54% | 12.15% | 10.41% |
| Annualized volatility | 16.21% | 13.40% | 13.80% | 14.34% | 13.38% | 16.08% |
| Sharpe ratio | 0.66 | 0.97 | 0.86 | 0.80 | 0.91 | 0.65 |
| Max drawdown | −28.54% | −23.90% | −24.04% | −25.82% | −24.04% | −27.88% |

## Layout

- `src/data.py` — load closes, log/simple returns, `vol_10`.
- `src/signals.py` — `compute_risk_adj_return`, `compute_slow_signal`.
- `src/stack_backtest.py` — Phase 1 engine; sub-strategies, rebalance, dynamic
  re-entry, combine. **Frozen — do not modify.**
- `src/metrics.py` — performance metrics.
- `src/benchmarks.py` — SPY buy-and-hold comparison.
- `src/overbought_filter.py` — Phase 2 overlay (load signal, build cash mask,
  apply filter).
- `src/individual_overbought.py` — Phase 2b overlay (analyze gate, SMA
  overbought flags, daily eligibility/selection/substitution, per-sub +
  combined returns). **Imports from Phase 1/2; modifies neither.**
- `src/divergence_filter.py` — Phase 3 overlay (load divergence percentile,
  build the IN/OUT cash mask, stack on the Phase 2 mask). **Imports from Phase
  1/2; modifies neither.**
- `src/divergence_signal.py` — recompute the divergence percentile from raw
  closes (point-in-time, expanding, no look-ahead);
  `scripts/build_divergence_signal.py` regenerates the CSV below.
- `data/overbought_signal.csv` — Phase 2 input (`date, market_ok`).
- `data/overbought_individual.csv` — Phase 2b input (uses `date, analyze`).
- `data/divergence_percentile.csv` — Phase 3 input (`date, divergence_pctl`);
  generated by `src/divergence_signal.py`.
- `docs/overbought_methodology.md` — how `market_ok` was derived.
- `docs/individual_overbought_methodology.md` — the analyze gate + individual
  overbought test (incl. the 6 imputed dates).
- `tests/` — data, signals, mechanics, overbought-filter, individual-overbought,
  divergence-filter, and divergence-signal tests; `test_phase1_frozen.py` (Phase 1 metrics
  snapshot) and `test_validation.py` (Phase 1 & Phase 2 daily-series snapshots),
  with fixtures under `tests/fixtures/`.
- `scripts/run_stack_backtest.py` — end-to-end run → figures + tables (incl. the
  Phase 2 filtered-vs-unfiltered and Phase 2b comparison outputs).

## Conventions

Python 3.11+, type hints, NumPy-style docstrings. No print statements in
library code (scripts may print). No external data downloads — only `data/`.
Each phase is a **purely additive overlay** — never modify a prior phase's code
or outputs (a frozen-snapshot test guards Phase 1). Do not reference the
`legacy-v1` branch. Run `pytest tests/ -v` before committing.
