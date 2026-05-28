# etf-momentum

Python reproduction of a tactical ETF allocation strategy ("Stack Portfolio")
with risk-adjusted momentum signals.

**Phase 1** reproduces an Excel backtest of the Stack Portfolio over a 43-ETF
universe (2015-12-07 → 2026-05-22): four parallel sub-strategies, each holding
up to five risk-adjusted-momentum names rebalanced every 20 trading days, with
daily in/out toggling and no compounding. See [CLAUDE.md](CLAUDE.md) for the
briefing and [RESULTS.md](RESULTS.md) for methodology and metrics.

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -v                      # all tests incl. the validation CI gate
python scripts/run_stack_backtest.py  # figures + tables under reports/
```

## Headline (reproduced vs. Excel target)

| Metric | Target | Reproduced |
|--------|--------|------------|
| Total return | 108.58% | 110.99% |
| Annualized return | 10.82% | 10.63% |
| Annualized volatility | 16.16% | 16.21% |
| Sharpe ratio | 0.67 | 0.66 |
| Max drawdown | −28.49% | −28.54% |

All metrics within ±3% relative tolerance.

## Layout

```
src/        data loading, signals, backtest engine, metrics
tests/      data / signal / mechanics tests + test_validation.py (CI gate)
scripts/    run_stack_backtest.py (end-to-end)
reports/    figures/ and tables/ (generated)
data/       raw closes + universe (inputs; do not modify)
```
