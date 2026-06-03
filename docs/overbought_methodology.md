# Overbought Market Filter — Methodology

## What this document covers

This document explains how the values in `data/overbought_signal.csv` were
determined. That file contains one binary value per trading day:

- `market_ok = 1` — the market is **not** overbought; the strategy is free to trade.
- `market_ok = 0` — the market is **overbought**; the strategy sits out (see the
  Phase 2 cash-window rule in `RESULTS.md` for how a `0` translates into cash days).

The signal is currently provided to the backtest as a **precomputed input**. This
document is its specification — it describes the logic used to generate the `0`s and
`1`s, so that the filter is transparent and auditable rather than an unexplained
column of numbers. A note on reproducibility is given at the end.

## Why an overbought filter exists

Markets occasionally reach stretched, overheated conditions from which further upside
is difficult and sharp pullbacks become more likely. A momentum strategy is
structurally most exposed to this risk: by construction it holds whatever has already
risen the most, which is precisely what tends to be most extended at a local top. The
overbought filter is a risk-management overlay designed to step aside during these
stretched conditions rather than press into them.

## The four trigger rules

The market is flagged overbought on a given day if **any one** of the following four
conditions is true (logical OR). Each rule looks at a different facet of "overheated."

### Rule 1 — Price stretch above moving averages (S&P 500)

Measures how far the S&P 500 index has run above its own trend. The market is
overbought if **either**:

- the SPX is more than **3% above** its 50-day **simple** moving average, **or**
- the SPX is more than **5% above** its 21-day **exponential** moving average.

Two horizons (medium-term simple, shorter-term exponential) catch different flavors of
price extension.

### Rule 2 — Breadth above the 10-day moving average (S&P 500)

Measures near-term participation across the index. The market is overbought when
**more than 85%** of S&P 500 member stocks are trading above their own 10-day moving
average. A reading this high means almost the entire index is extended at once —
a breadth extreme.

### Rule 3 — RSI breadth (Nasdaq 100)

Measures momentum-oscillator froth concentrated in the large-cap growth names. The
market is overbought when **more than 20%** of Nasdaq 100 member stocks have a
Relative Strength Index (RSI) **above 70**. RSI above 70 is a conventional
single-name overbought threshold; the rule fires when an unusually large share of the
index is simultaneously in that zone.

### Rule 4 — Self-referential momentum check (the strategy's own returns)

Flags overbought when the strategy's *own* recent performance has run hot relative to
its historical norm. Unlike Rules 1–3, this uses only the momentum portfolio's own
return series. It is computed as follows:

1. Take the portfolio's daily returns.
2. For each day, compute the **trailing 10-day cumulative return** (the sum of the
   last 10 daily returns).
3. Compute the **5-day average of that 10-day cumulative return** — i.e., average the
   five most recent overlapping 10-day cumulative returns. This smoothing damps
   single-day noise so one unusually strong day does not trip the filter on its own.
4. Compute the **historical mean** and **historical standard deviation** of the 10-day
   cumulative return series.
5. The market is flagged overbought when the smoothed figure from step 3 exceeds:

   ```
   historical mean  +  (1/3) × historical standard deviation
   ```

In plain terms: when the strategy's recent, smoothed run is more than a third of a
standard deviation hotter than its typical run, it is treated as overextended.

## The post-crash suppression rule

A crucial qualification: markets are *naturally* and *persistently* overbought during
the recovery that follows a sharp decline. The violent bounce off a bottom pegs every
overbought indicator — but that is a bullish condition, not a reason to sit out. Acting
on overbought readings during a recovery would mean stepping aside during the strongest
part of the rebound.

To avoid this, the four rules above are **switched off entirely** for a window
following a significant drawdown. The length of that window scales with the depth of
the decline:

| S&P 500 drawdown | Overbought filter disabled for |
|------------------|--------------------------------|
| 5%               | 1.0 month                      |
| 7.5%             | 1.5 months                     |
| 10%              | 2.0 months                     |
| 15%              | 2.5 months                     |
| 20%              | 3.0 months                     |
| 25%              | 3.5 months                     |
| 30%              | 4.0 months                     |
| 35%              | 4.5 months                     |

**Worked example (COVID, 2020).** The S&P 500 fell roughly 35% into the March 2020
low. During the first weeks of the recovery the market repeatedly hit overbought
levels — which was entirely expected given the severity of the preceding crash.
Applying the overbought filter there would have forced the strategy to cash during the
sharpest part of the rebound. Under the table above, a ~35% drawdown disables the
filter for ~4.5 months, allowing the strategy to participate fully in the recovery.

## How the rules map to the signal file

For each trading day, the `market_ok` value in `data/overbought_signal.csv` is set as
follows:

- If the day falls inside a post-crash suppression window → `market_ok = 1`
  (overbought rules ignored).
- Otherwise, if **any** of Rules 1–4 fires → `market_ok = 0` (overbought).
- Otherwise → `market_ok = 1`.

## Reproducibility note

This signal is provided as a precomputed input rather than regenerated from raw market
data inside this repository, for a specific and honest reason: **Rules 2 and 3 require
index-constituent breadth data** — the percentage of S&P 500 members above their
10-day moving average, and the percentage of Nasdaq 100 members with RSI above 70.
These require per-constituent price histories for roughly 600 underlying stocks
(sourced originally via Bloomberg), which are not redistributed in this repository.

Of the four rules, **Rule 4 is fully reproducible from data already in this repo**, as
it depends only on the strategy's own return series. Rules 1–3 and the drawdown-based
suppression depend on external SPX/NDX index and constituent data. The signal file
therefore represents the complete filter as originally computed; this document
specifies the logic behind it so the methodology is transparent and verifiable even
where the underlying constituent data is not bundled here.
