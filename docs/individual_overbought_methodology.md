# Individual-ETF Overbought Filter — Methodology (Phase 2b)

## What this document covers

This document explains the **analyze gate** consumed by Phase 2b — the
`analyze` column in `data/overbought_individual.csv` — and the individual-ETF
overbought test that Phase 2b computes from raw closes. Phase 2b is a purely
additive overlay: it never modifies the Phase 1 engine, the Phase 2 market
filter, or any of their outputs.

The file `data/overbought_individual.csv` has columns
`date, market_overbought, cancel, overbought_net, analyze`. **Only `date` and
`analyze` are used by Phase 2b.** The other three columns are carried along
from the source workbook for provenance but are not read by the code.

- `analyze = 1` — the individual-ETF overbought screen is **active** that day:
  names that are individually overbought are excluded from selection.
- `analyze = 0` — the screen is **off** that day: selection falls back to
  "top-5 by slow signal among strictly-positive names," exactly like the
  unscreened ranking.

The gate is an exogenous, precomputed input. Phase 2b consumes it as-is and
does **not** recompute it.

## The individual-ETF overbought test

Independent of the analyze gate, Phase 2b computes — from the raw closes in
`data/closes_wide.csv` — whether each ETF is individually overbought on each
day. An ETF is overbought on day `t` when **either** (logical OR):

```
close_t  >=  1.05 * SMA20_t        # 20-day simple moving average of its close
close_t  >=  1.07 * SMA50_t        # 50-day simple moving average of its close
```

Both comparisons are inclusive (`>=`), and both moving averages are **simple**
(equal-weighted) averages of that ETF's own close. An ETF without a full 20- or
50-day window has a NaN SMA; comparisons against NaN are `False`, so such a name
is simply not flagged (no `fillna`, no special-casing). The exclusion has **no
timer**: a name is excluded only on the days it is overbought and is eligible
again the moment it cools off — substitution and swap-back fall out of the daily
recompute.

The screen only *acts* on days the analyze gate is active (`analyze == 1`). On
`analyze == 0` days the overbought flags are computed but ignored.

## Optional cooldown timer

The selection logic exposes a `cooldown_days` parameter (default `0`). With
`cooldown_days = 0` the screen is the no-timer rule above. With
`cooldown_days = N > 0`, a name flagged overbought on a decision day is barred
for the **next N trading days** as well, even if it cools sooner. A fresh flag
while the name is still barred restarts the N-day count, and overlapping bars
merge — equivalently, the name is barred whenever it was flagged on any of the
previous N decision days. Because flags are gated by the analyze gate, no new bar
starts on `analyze == 0` days, but an existing bar keeps counting through them.
The minimum one-day offset preserves the no-look-ahead property (a day's bar
depends only on flags strictly before it). The shipped `cooldown_days = 0` keeps
the base Phase 2b behavior; a `cooldown_days = 5` variant is reported in
`RESULTS.md` for comparison.

## The 6 imputed analyze dates

Six consecutive rows in `data/overbought_individual.csv` are **imputed** rather
than sourced directly:

```
2025-09-05, 2025-09-08, 2025-09-09, 2025-09-10, 2025-09-11, 2025-09-12
```

All six carry `analyze = 1`. These dates correspond to a short gap in the
underlying source data; the values were filled to keep the gate defined across
the full trading calendar. Because Phase 2b reads only `date` and `analyze`,
the imputation affects only whether the individual screen is active on those six
days (it is). Their effect on headline metrics is negligible, but the imputation
is flagged here for transparency and auditability.

## Edge handling

- Trading days absent from `overbought_individual.csv` are treated as
  `analyze == 0` (screen off) — the neutral default that never excludes a name
  on the basis of a missing gate value. (In the shipped data every strategy
  trading day within the gate's range is present, so this fallback is exercised
  only outside that range.)
- No look-ahead: the roster held on day `t` is chosen from the slow signal,
  overbought flags, and analyze gate observed at `t-1` (decided at the prior
  close, accruing on `t`), mirroring Phase 1's `roster_lag = 1`.
