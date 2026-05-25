# Data

## Files

| File | Description |
|---|---|
| `etf_panel.parquet` | Long-format daily panel. Columns: `date, ticker, category, close, return, log_return, vol_10d_ann, signal_ret_over_vol, sheet_name`. ~128k rows. Primary data source for all analysis. |
| `closes_wide.csv` | Wide-format price matrix: dates × 43 tickers. Convenient for Excel/Numbers inspection and vectorized cross-sectional ops. |
| `universe.csv` | The 43-ETF universe map: sheet name → ticker → category → coverage dates. |
| `splits_applied.csv` | Audit log of stock-split back-adjustments applied during cleaning. |

## Coverage

- 43 ETFs spanning global equity, US sectors, US styles, single countries, regions, REITs, and US fixed income.
- Daily frequency, July 3, 2014 → May 22, 2026.
- ~2,990 observations per ticker. Fully balanced panel, no gaps.

## Provenance

Original prices pulled from Bloomberg Terminal (`PX_LAST` field) and stored in a single multi-sheet `.xlsx` workbook. The cleaning pipeline:

1. Extracts daily `(date, close)` per sheet, normalizing the WORLD sheet (which had the index in columns A:B and the ETF in W:X) to use VT ETF data.
2. Coerces Excel-serial integer dates back to proper timestamps.
3. Recomputes daily returns, log returns, 10-day annualized volatility, and the original Excel return/vol signal in Python to match the source workbook's formulas exactly (verified against SPY 2014-11-19).
4. Detects unadjusted stock splits via single-day return magnitude (threshold |r| > 0.20) and back-adjusts pre-split prices.

## Known split adjustments

| Ticker | Date | Ratio | Notes |
|---|---|---|---|
| IWF | 2025-12-31 | 4-for-1 | Pre-split prices divided by 4 for continuity. Without this, the unadjusted -75% one-day "return" would dominate any momentum-based signal. |
