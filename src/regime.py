"""Market regime filters for conditional execution.

Cross-sectional momentum is known to fail badly during sharp market reversals
and high-correlation crisis regimes. This module provides simple filters that
identify benign regimes (where momentum tends to work) and toggle the
strategy off — to cash — when those regimes are not in force.

Two filters are implemented:

* ``trend_filter`` — long-only when broad market is above its N-day moving
  average. Inspired by Faber (2007), "A Quantitative Approach to Tactical
  Asset Allocation."
* ``vol_filter`` — long-only when realized market volatility is below an
  expanding-window percentile. Reduces exposure when risk is elevated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trend_filter(
    panel: pd.DataFrame,
    market_ticker: str = "spy",
    lookback_days: int = 200,
) -> pd.Series:
    """Boolean series: True when market close is above its trailing MA.

    The filter is evaluated daily but the practical effect is regime-conditional
    holding: when False, the strategy moves to cash at the next rebalance.

    Args:
        panel: Long-format ETF panel.
        market_ticker: Which ticker to use as the market proxy.
        lookback_days: MA window length (200 is conventional).

    Returns:
        Boolean Series indexed by date.
    """
    market = (
        panel[panel["ticker"] == market_ticker]
        .set_index("date")["close"]
        .sort_index()
    )
    ma = market.rolling(lookback_days).mean()
    return (market > ma).rename(f"trend_above_{lookback_days}d_ma")


def vol_filter(
    panel: pd.DataFrame,
    market_ticker: str = "spy",
    vol_window: int = 21,
    expanding_pct_threshold: float = 0.80,
) -> pd.Series:
    """Boolean series: True when market vol is below its expanding-window threshold.

    Uses an expanding-window quantile so the filter is purely backward-looking
    (no peeking at future volatility distribution).

    Args:
        panel: Long-format ETF panel.
        market_ticker: Market proxy ticker.
        vol_window: Realized-vol window in trading days.
        expanding_pct_threshold: Percentile cutoff. If today's vol exceeds the
            expanding-window quantile, the filter returns False (risk-off).

    Returns:
        Boolean Series indexed by date.
    """
    market = panel[panel["ticker"] == market_ticker].set_index("date").sort_index()
    realized = market["log_return"].rolling(vol_window).std() * np.sqrt(252)
    # Expanding-window threshold avoids look-ahead
    threshold = realized.expanding(min_periods=vol_window * 4).quantile(expanding_pct_threshold)
    return (realized < threshold).fillna(False).rename(
        f"vol_below_{int(expanding_pct_threshold*100)}p"
    )


def combined_filter(
    panel: pd.DataFrame,
    use_trend: bool = True,
    use_vol: bool = False,
    market_ticker: str = "spy",
    trend_lookback: int = 200,
    vol_window: int = 21,
    vol_threshold: float = 0.80,
) -> pd.Series:
    """AND-combine the trend and vol filters into a single regime gate."""
    gate = pd.Series(True, index=panel["date"].unique())
    gate = gate.sort_index()
    if use_trend:
        t = trend_filter(panel, market_ticker, trend_lookback)
        gate = gate.reindex(t.index).fillna(False) & t.fillna(False)
    if use_vol:
        v = vol_filter(panel, market_ticker, vol_window, vol_threshold)
        gate = gate.reindex(v.index).fillna(False) & v.fillna(False)
    return gate.rename("regime_gate")
