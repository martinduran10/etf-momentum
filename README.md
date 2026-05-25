# ETF Momentum

A reproducible falsification study of cross-sectional momentum across a 43-ETF global universe (regions, sectors, styles, fixed income), with daily Bloomberg data from July 2014 to May 2026.

The project takes a classic factor strategy and subjects it to the full suite of stress tests a quant researcher should run *before* deploying anything: parameter sensitivity, regime overlays, walk-forward out-of-sample validation, factor regression with HAC standard errors, block-bootstrap significance testing, cost sensitivity, and a dollar-neutral long/short variant.

**For the full writeup of findings, see [RESULTS.md](RESULTS.md).**

## Headline findings

| Test | Result |
|---|---|
| Baseline 6-1 momentum vs SPY | α = **-4.88%** ann., t = -2.03, **p = 0.042** (statistically significant negative alpha) |
| Best parameter sweep (50 grid) | Sharpe 0.55, 95% CI [+0.03, +1.14] — marginal significance, by construction overfit |
| Walk-forward OOS | Stitched Sharpe **0.13**, 95% CI [-0.50, +0.78] (includes zero) |
| Long/short dollar-neutral factor | α = +0.29% ann., **p = 0.90**, Sharpe 0.10 [-0.48, +0.71] |
| Transaction-cost break-even vs SPY | None — gross strategy CAGR caps at 7.7% vs SPY's 11.9% |

Across every cut, there is no statistical evidence of a tradable cross-sectional momentum edge in this ETF universe and sample. The project is as much about the methodology of *honestly testing* a strategy as it is about the strategy itself.

## Repo structure

```
etf-momentum/
├── RESULTS.md                   # Full research writeup with all charts and tables
├── data/                        # Cleaned Bloomberg panel (see data/README.md)
├── src/                         # Library code
│   ├── data.py                  # Panel loading and pivoting
│   ├── signals.py               # Raw and risk-adjusted momentum signals
│   ├── backtest.py              # Long-only and long/short engine with regime gating
│   ├── metrics.py               # Sharpe, Sortino, Calmar, drawdown, hit rate
│   ├── regime.py                # 200-day MA trend filter, vol filter
│   ├── sensitivity.py           # Parameter grid search
│   ├── analysis.py              # Factor regression (HAC), bootstrap CIs, cost sweep
│   └── visualization.py         # Publication-quality matplotlib helpers
├── scripts/
│   └── generate_results.py      # End-to-end research driver — regenerates everything
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory analysis
├── tests/                       # 26 pytest tests
└── reports/
    ├── figures/                 # PNG charts
    └── tables/                  # CSV summary tables
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/etf-momentum.git
cd etf-momentum
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
pytest tests/ -v                   # should show 26 passing
```

Or with the Makefile:

```bash
make install
make test
make results
```

`make results` runs `scripts/generate_results.py` end-to-end, regenerating every figure and table in `reports/`.

## Data provenance

Daily closing prices were originally pulled from Bloomberg Terminal (`PX_LAST`) for the 43-ticker universe. The raw `.xlsx` workbook contained 45 sheets with embedded formulas computing daily returns, log returns, 10-day rolling volatility, and a return/volatility signal. The cleaning pipeline parses these into a Pandas panel, coerces serial-date integers to proper timestamps, and back-adjusts one unadjusted IWF 4-for-1 stock split on 2025-12-31. See `data/README.md` for full details.

## Methodology notes

- **No look-ahead.** Signal at rebalance date `t` is computed only from data ending at `t`; positions earn returns from `t+1` onward. Enforced by automated tests.
- **HAC standard errors.** Factor regressions use Newey-West with 10-day lag length to account for residual autocorrelation introduced by monthly rebalancing.
- **Block bootstrap.** Sharpe ratio confidence intervals use circular block bootstrap (Politis & Romano 1992) with 21-day blocks and 2,000 replicates. Block length matches the typical holding period; circular wrap-around gives every observation equal probability of inclusion.
- **Walk-forward.** Parameters chosen on a rolling 4-year training window, evaluated on the immediately following 1.5-year out-of-sample window, rolled forward by 1.5 years.

## References

- Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers." *Journal of Finance*.
- Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere." *Journal of Finance*.
- Daniel & Moskowitz (2016), "Momentum Crashes." *Journal of Financial Economics*.
- Newey & West (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*.
- Politis & Romano (1992), "A Circular Block-Resampling Procedure for Stationary Data."

## Author

Martin Duran. Junior, Business Analytics (BSBA), University of Miami Herbert Business School (Spring 2027). Prior internship in financial data modeling and backtesting at Altex Asset Management.

## License

MIT — see [LICENSE](LICENSE).
