# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install        # pip install -r requirements.txt
make test           # pytest tests/ -v   (26 tests, expected to all pass)
make results        # python scripts/generate_results.py — regenerates every figure/table in reports/
make clean          # removes caches AND deletes reports/figures/*.png + reports/tables/*.csv
```

Run a single test:
```bash
pytest tests/test_signals.py::test_momentum_signal_has_no_lookahead -v
```

CI runs `pytest tests/ -v --tb=short` on Python 3.11 and 3.12 (see [.github/workflows/tests.yml](.github/workflows/tests.yml)).

## Architecture

The codebase is a research pipeline, not a service. Data flows one direction:

```
data/etf_panel.parquet  →  signals.py  →  backtest.py  →  metrics.py / analysis.py  →  visualization.py
                                              ↑
                                         regime.py (optional gate)
```

[scripts/generate_results.py](scripts/generate_results.py) is the canonical driver wiring all of this together end-to-end. Anything reported in [RESULTS.md](RESULTS.md) is reproducible by running that one script.

### Data model

The canonical data structure is the **long-format panel** loaded by `src.data.load_panel()` — one row per `(date, ticker)` with columns `date, ticker, category, close, return, log_return, vol_10d_ann, signal_ret_over_vol, sheet_name`. Every module accepts and returns this shape. Convert to wide (dates × tickers) with `src.data.to_wide(panel, column=...)` only when you need a price/return matrix for vectorized cross-sectional ops.

The panel ships pre-cleaned in `data/etf_panel.parquet`: Bloomberg `PX_LAST` for 43 ETFs, July 2014 → May 2026, with an IWF 4-for-1 split (2025-12-31) already back-adjusted. The Excel cleaning pipeline is not in this repo — treat the parquet as the immutable source of truth.

### No-look-ahead is a hard invariant

This is the central methodological commitment and it's enforced in tests, not just convention:

- Signals at date `t` use only data ending at `t`. `momentum_total_return` shifts prices by `skip_months × 21` and `lookback_months × 21` trading days.
- `run_backtest` computes target weights at the rebalance date, then `.shift(1)` so positions earn returns starting `t+1`.
- Regime filters use expanding-window quantiles, never full-sample percentiles.
- `tests/test_signals.py::test_momentum_signal_has_no_lookahead` will fail if a signal becomes valid before ~6 months of per-ticker history exist.

When adding new signals or filters, preserve this. The test suite is the contract.

### Day-count conventions

`TRADING_DAYS_PER_YEAR = 252` and `TRADING_DAYS_PER_MONTH = 21` are defined in both `src/signals.py` and `src/metrics.py`. Annualization uses 252 throughout. Rebalance frequency strings follow pandas `Grouper` (`"W"`, `"ME"` month-end, `"QE"` quarter-end) — note `"ME"`/`"QE"`, not the deprecated `"M"`/`"Q"`.

### Statistical machinery

Three pieces in `src/analysis.py` are deliberately non-default and should not be "simplified" away:

- **HAC standard errors** (Newey-West, `cov_type="HAC"`) in `regress_against_market`. Strategy returns are serially correlated by construction (monthly rebalance → overlapping holding periods), so naive OLS standard errors are too tight.
- **Circular block bootstrap** (Politis & Romano 1992) in `bootstrap_sharpe_ci`, default 21-day blocks. IID bootstrap would give falsely-narrow Sharpe CIs for the same autocorrelation reason.
- **Annual cost drag** in `cost_sensitivity` is computed from per-rebalance turnover summed and annualized — *not* from a daily-average of `cost_series`, which would dilute with zero-cost non-rebalance days and badly understate the drag.

### Backtest engine surface

`run_backtest(panel, signal_col, ...) -> BacktestResult`:

- `n_short=0` (default) → equal-weight long-only top-N. `n_short>0` → dollar-neutral long/short with each leg weighted at `±1/max(n_long, n_short)`.
- `regime_gate` is an optional boolean `Series` indexed by date. When `False` on a rebalance date, the strategy goes flat for that period (positions remain zero, `invested_flags` records it, and `invested_pct` reports the fraction of rebalance dates the strategy was actually in the market).
- Costs are applied as `cost_bps_per_side × turnover` where turnover = `Σ|Δw|` at each rebalance.

### Output discipline

`scripts/generate_results.py` writes numbered outputs (`01_baseline_*`, `02_*`, …) into `reports/figures/` and `reports/tables/`. The numbering mirrors the section order in `RESULTS.md`, so when adding a new experiment, pick a number that fits the narrative position and update `RESULTS.md` to reference it.

v2 phase outputs use named (not numbered) prefixes: `regime_hmm_overlay.png`, `regime_hmm_summary.csv`, etc. — they're driven by per-phase scripts rather than the master `generate_results.py`.

## Strategy versions

- **v1 — vanilla cross-sectional momentum** (shipped). Falsification study with HAC alpha, block-bootstrap CIs, walk-forward OOS, and L/S variant. Conclusion documented in [RESULTS.md](RESULTS.md): no statistically significant tradable edge in this universe and sample.
- **v2 — regime-adaptive (in progress)**. Hypothesis: momentum behaves asymmetrically across market regimes, and conditioning on the detected regime can produce defensible alpha where vanilla momentum cannot. Tracked in [RESULTS_v2.md](RESULTS_v2.md).
  - *Phase 1 (current)*: HMM regime detection — [src/regime_hmm.py](src/regime_hmm.py), driver [scripts/generate_regime_overlay.py](scripts/generate_regime_overlay.py). Descriptive characterization only, no alpha claim.
  - *Phase 2+ (planned)*: regime-conditional signals, expanding-window HMM fit, vol targeting, dashboard.

## Regime convention

`src.regime_hmm.fit_hmm_regimes` returns integer regime labels with a strict ordering invariant: **regime 0 = lowest mean weekly return (bear), regime `n_regimes - 1` = highest (bull)**. The fitted model's `means_`, `_covars_`, `transmat_`, and `startprob_` are permuted in place to match the relabeled state numbering — so a downstream caller using `model.predict(...)` directly will get labels in the same ordering as the returned `regime_series`. Test invariant: `tests/test_regime_hmm.py::test_regime_ordering_invariant` will fail if this ordering is ever broken.

Daily expansion via `expand_regimes_to_daily` is forward-fill only — the label on date X reflects the most recent week-end ≤ X. The HMM itself is currently fit on the full series (Viterbi peek); this is acceptable for Phase 1's descriptive role but must be replaced with an expanding-window fit before any regime-driven trading decision is evaluated.
