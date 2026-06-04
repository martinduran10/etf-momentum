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

## Layout

```
src/        data loading, signals, backtest engine, metrics
tests/      data / signal / mechanics tests + test_validation.py (CI gate)
scripts/    run_stack_backtest.py (end-to-end)
reports/    figures/ and tables/ (generated)
data/       raw closes + universe (inputs; do not modify)
```
