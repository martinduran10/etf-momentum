# etf-momentum

Python reproduction of a tactical ETF allocation strategy ("Stack Portfolio")
with risk-adjusted momentum signals and tactical risk filters.

**Phase 1** reproduces an Excel backtest of the Stack Portfolio over a 43-ETF
universe (2015-12-07 → 2026-05-22): four parallel sub-strategies, each holding
up to five risk-adjusted-momentum names rebalanced every 20 trading days, with
daily in/out toggling and no compounding. See [CLAUDE.md](CLAUDE.md) for the
briefing and [RESULTS.md](RESULTS.md) for methodology and metrics.

**Phase 2** layers an overbought-market filter on top of Phase 1 as a purely
additive overlay: an exogenous binary signal (`data/overbought_signal.csv`)
flags days the broad market is overbought, and the strategy sits in cash on the
four trading days following each flag (T+3 … T+6, with overlapping windows
merging). Phase 1's engine and outputs are left byte-for-byte unchanged. See
[RESULTS.md](RESULTS.md) for the full rule and edge cases.

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -v                      # all tests incl. the validation CI gate
python scripts/run_stack_backtest.py  # figures + tables under reports/
```

## Phase 1 — Headline (reproduced vs. Excel target)

| Metric | Target | Reproduced |
|--------|--------|------------|
| Total return | 108.58% | 110.99% |
| Annualized return | 10.82% | 10.63% |
| Annualized volatility | 16.16% | 16.21% |
| Sharpe ratio | 0.67 | 0.66 |
| Max drawdown | −28.49% | −28.54% |

All metrics within ±3% relative tolerance.

## Phase 2 — Overbought filter (unfiltered vs. filtered)

| Metric | Unfiltered (Phase 1) | Filtered (Phase 2) |
|--------|----------------------|--------------------|
| Total return | 110.99% | 136.18% |
| Annualized return | 10.63% | 13.05% |
| Annualized volatility | 16.21% | 13.40% |
| Sharpe ratio | 0.66 | 0.97 |
| Max drawdown | −28.54% | −23.90% |
| % of days in cash | — | 36.96% |

The filter improves every dimension — higher return and Sharpe at lower
volatility and drawdown — by sidestepping clusters of bad days rather than
amplifying good ones. It closely tracks the deck's published Phase II
(134.94% / Sharpe 1.01).

## Phase 2b — Individual-ETF overbought filter

**Phase 2b** adds a second additive overlay that **rebalances daily**: each
sub-strategy re-ranks all 43 ETFs every day and holds the top five *eligible*
names, where eligibility is `slow_signal > 0` and — on days the exogenous
analyze gate (`data/overbought_individual.csv`) is active — *not* individually
overbought (close ≥ 1.05 × its 20-day SMA **or** ≥ 1.07 × its 50-day SMA).
Screened-out names yield their slot to the next ranked name. By default there is
no timer (a name reclaims its slot the moment it cools off); an optional
`cooldown_days` bars a flagged name for N more trading days. The Phase 2 market
mask is applied last, unchanged. The table below adds the 5-day-cooldown variant
and two attribution series.

| Metric | Phase 1 | Phase 1+2 | Phase 2b | Phase 2b + 5d cooldown | Daily-rebal + mask (screen off) | Daily-rebal (screen off) |
|--------|---------|-----------|----------|------------------------|---------------------------------|--------------------------|
| Total return | 110.99% | 136.18% | 124.30% | 120.39% | 126.80% | 108.61% |
| Annualized return | 10.63% | 13.05% | 11.91% | 11.54% | 12.15% | 10.41% |
| Annualized volatility | 16.21% | 13.40% | 13.80% | 14.34% | 13.38% | 16.08% |
| Sharpe ratio | 0.66 | 0.97 | 0.86 | 0.80 | 0.91 | 0.65 |
| Max drawdown | −28.54% | −23.90% | −24.04% | −25.82% | −24.04% | −27.88% |

The highest Sharpe in the table is Phase 1+2 (0.97). Over this window neither the
individual screen nor the 5-day cooldown raises risk-adjusted return: Phase 2b's
0.86 is below the 0.91 of daily-rebal+mask (screen off), and the cooldown is
lower than no-timer Phase 2b on every metric shown. The two right-hand columns
attribute the effect — the daily rebalance alone is near-neutral (0.65), and the
market mask carries most of the gain (0.91). See [RESULTS.md](RESULTS.md) and
[docs/individual_overbought_methodology.md](docs/individual_overbought_methodology.md)
(six analyze dates, 2025-09-05 … 2025-09-12, are imputed).

## Layout

```
src/        data loading, signals, backtest engine, metrics
tests/      data / signal / mechanics tests + test_validation.py (CI gate)
scripts/    run_stack_backtest.py (end-to-end)
reports/    figures/ and tables/ (generated)
data/       raw closes + universe (inputs; do not modify)
```
