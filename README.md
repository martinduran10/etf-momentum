# etf-momentum

A reproducible study of how tactical **risk overlays** affect the drawdown and
risk-adjusted return of a momentum portfolio of 43 global ETFs (2015–2026).

Momentum's structural weakness is crash risk: by construction it holds whatever
has already run the most, which is exactly what tends to be most extended near a
top. This project starts from a faithfully reproduced momentum backtest (Phase 1)
and asks one question — **how much can simple, rules-based overlays reduce that
crash risk, and at what cost?** Three overlays are tested: an overbought-market
filter (Phase 2), an individual-name overbought screen (Phase 2b), and a
momentum-divergence filter (Phase 3). Each is *purely additive* — it masks the
layer beneath it without altering a line of it — a discipline enforced by
frozen-snapshot regression tests.

### Headline

Stacking the overlays over Dec 2015 – May 2026:

| | Phase 1 (base) | Phase 2 (+ market filter) | Phase 3 (+ divergence filter) |
|---|---|---|---|
| Max drawdown | −28.5% | −23.9% | **−18.3%** |
| Sharpe ratio | 0.66 | 0.97 | **1.31** |
| Days in cash | — | 37% | 43% |

The overlays cut maximum drawdown by roughly a third and raise the Sharpe ratio,
by sidestepping clusters of bad days rather than amplifying good ones — the
strategy sits in cash ~43% of the time. Not every overlay helps: the
individual-name screen (Phase 2b) is reported as a **negative result**, and the
repo documents why it does not improve on the market filter.

> **Read these as an in-sample study, not a validated strategy.** The figures are
> gross of transaction costs, fit over a single market regime, and not yet
> walk-forward-validated. The divergence signal is recomputed in-repo with a
> no-look-ahead test; the Phase 2 signal is still an exogenous input. The full
> list of caveats is in [Limitations](RESULTS.md#limitations).

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -v                      # all tests incl. frozen-snapshot regression
python scripts/run_stack_backtest.py  # figures + tables under reports/
```

## Phase 1 — Headline (Excel reproduction, historical note)

The strategy is no longer validated against the deck's published numbers; phases
are reported empirically and prior phases are pinned by frozen-snapshot tests.
This table is kept only to document the original reproduction provenance.

| Metric | Excel (published) | Reproduced |
|--------|-------------------|------------|
| Total return | 108.58% | 110.99% |
| Annualized return | 10.82% | 10.63% |
| Annualized volatility | 16.16% | 16.21% |
| Sharpe ratio | 0.67 | 0.66 |
| Max drawdown | −28.49% | −28.54% |

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

## Phase 3 — Divergence filter

**Phase 3** stacks a third additive overlay on the main line (Phase 1 + Phase 2),
driven by a divergence percentile (`data/divergence_percentile.csv`, the spread
between the top-5 and bottom-5 ETFs on the momentum signal). A market-level state
machine sits the strategy in cash on an edge-triggered upward cross through the
0.85 percentile and re-enters when either the percentile normalizes (< 0.85) or
the Phase 1 base curve has corrected ≥ 8 points from its peak since the exit (all
decisions T+1, no look-ahead). Phase 3 is cash whenever the Phase 2 mask **or**
the divergence mask says cash. The percentile, once an exogenous CSV, is now
reproduced in-repo from raw closes ([src/divergence_signal.py](src/divergence_signal.py))
point-in-time with no look-ahead.

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| Total return | 110.99% | 136.18% | 167.85% |
| Annualized return | 10.63% | 13.05% | 16.08% |
| Annualized volatility | 16.21% | 13.40% | 12.25% |
| Sharpe ratio | 0.66 | 0.97 | 1.31 |
| Max drawdown | −28.54% | −23.90% | −18.26% |
| % of days in cash | — | 36.96% | 43.19% |

The divergence overlay adds 6.24 percentage points of cash days beyond Phase 2
(43.19% total; overbought-only 32.85%, divergence-only 6.24%, both 4.11%). Over
this window it raises total return and Sharpe at lower volatility **and** a
materially shallower maximum drawdown (−23.90% → −18.26%). See
[RESULTS.md](RESULTS.md) for the full state-machine rule.

## Layout

```
src/        data loading, signals, backtest engine, metrics
tests/      data / signal / mechanics tests + frozen-snapshot regression
scripts/    run_stack_backtest.py (end-to-end)
reports/    figures/ and tables/ (generated)
data/       raw closes + universe (inputs; do not modify)
```
