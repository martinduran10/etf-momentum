"""Statistical analysis helpers.

Three capabilities:

* ``regress_against_market`` — single-factor CAPM-style regression of strategy
  returns on a market proxy. Returns annualized alpha, beta, t-statistics
  (with HAC standard errors that account for autocorrelation), and R².

* ``bootstrap_sharpe_ci`` — circular block bootstrap confidence interval on
  the Sharpe ratio. Block sampling preserves the autocorrelation structure
  of daily returns; the block length is chosen to match the holding-period
  serial dependence introduced by monthly rebalancing.

* ``cost_sensitivity`` — sweep over transaction-cost levels and report
  CAGR, Sharpe, and the per-rebalance cost drag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .backtest import run_backtest
from .metrics import annualized_return, sharpe_ratio, TRADING_DAYS_PER_YEAR


@dataclass
class FactorRegression:
    """Result of a single-factor regression."""

    alpha_daily: float
    alpha_annualized: float
    beta: float
    alpha_t_stat: float
    alpha_p_value: float
    beta_t_stat: float
    r_squared: float
    n_obs: int
    summary_text: str


def regress_against_market(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    rf_daily: float = 0.0,
    drop_zero_strategy_days: bool = False,
    hac_lags: int = 5,
) -> FactorRegression:
    """Single-factor regression: r_strat - rf = α + β·(r_mkt - rf) + ε.

    Standard errors use HAC (Newey-West) to account for residual
    autocorrelation, which is common in strategy returns due to monthly
    rebalancing creating overlapping holding periods.

    Args:
        strategy_returns: Daily strategy returns.
        market_returns: Daily market proxy returns (e.g., SPY).
        rf_daily: Risk-free rate per day. Set to 0 for excess-of-cash.
        drop_zero_strategy_days: If True, exclude days where the strategy
            holds zero positions (e.g., when a regime gate is off). Useful
            for isolating the alpha/beta of the *invested* portion only.
        hac_lags: Lag length for HAC standard errors. ~5 days is a
            reasonable default for daily returns with weekly seasonality;
            longer for strategies with strong monthly autocorrelation.

    Returns:
        ``FactorRegression`` with alpha (daily + annualized), beta, t-stats,
        the alpha p-value, R² and the full summary text.
    """
    df = pd.concat(
        [strategy_returns.rename("strat"), market_returns.rename("mkt")],
        axis=1,
        sort=True,
    ).dropna()

    if drop_zero_strategy_days:
        df = df[df["strat"].abs() > 1e-12]

    if len(df) < 20:
        raise ValueError(f"Need at least 20 observations for regression, got {len(df)}")

    y = df["strat"] - rf_daily
    X = sm.add_constant(df["mkt"] - rf_daily)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    alpha_daily = float(model.params.iloc[0])
    beta = float(model.params.iloc[1])
    return FactorRegression(
        alpha_daily=alpha_daily,
        alpha_annualized=alpha_daily * TRADING_DAYS_PER_YEAR,
        beta=beta,
        alpha_t_stat=float(model.tvalues.iloc[0]),
        alpha_p_value=float(model.pvalues.iloc[0]),
        beta_t_stat=float(model.tvalues.iloc[1]),
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
        summary_text=str(model.summary()),
    )


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 2000,
    block_length: int = 21,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Circular block bootstrap CI for the Sharpe ratio.

    Daily strategy returns are autocorrelated by construction: between
    monthly rebalances, the same positions are held, so consecutive daily
    returns share underlying exposures. An IID bootstrap ignores this and
    produces falsely-tight CIs.

    The circular block bootstrap (Politis & Romano, 1992) samples contiguous
    blocks of length ``block_length`` and wraps around end-of-series, which
    preserves serial dependence within blocks while giving every observation
    equal probability of inclusion.

    Block length of ~21 days (one trading month) matches the typical
    holding period of a monthly-rebalanced strategy: longer blocks would
    waste sample size; shorter blocks would underrepresent within-month
    serial structure.

    Args:
        returns: Daily return series.
        n_bootstrap: Number of bootstrap replicates.
        block_length: Block size in trading days.
        confidence: Confidence level (e.g., 0.95).
        rng_seed: RNG seed for reproducibility.

    Returns:
        Dict with ``sharpe``, ``ci_low``, ``ci_high``, ``ci_width``,
        ``n_bootstrap`` and ``includes_zero`` (handy for hypothesis tests).
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < block_length * 4:
        return {
            "sharpe": float(sharpe_ratio(returns)),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_width": float("nan"),
            "n_bootstrap": 0,
            "includes_zero": True,
        }

    rng = np.random.default_rng(rng_seed)
    n_blocks = int(np.ceil(n / block_length))
    # Circular bootstrap: wrap around the end of the series
    r_circ = np.concatenate([r, r[: block_length - 1]])  # pad for wrap

    sharpes = np.empty(n_bootstrap, dtype=float)
    sqrt_year = np.sqrt(TRADING_DAYS_PER_YEAR)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([r_circ[s : s + block_length] for s in starts])[:n]
        sd = sample.std(ddof=1)
        sharpes[i] = (sample.mean() / sd) * sqrt_year if sd > 0 else np.nan

    alpha = 1 - confidence
    low_q, high_q = np.nanpercentile(sharpes, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {
        "sharpe": float(sharpe_ratio(returns)),
        "ci_low": float(low_q),
        "ci_high": float(high_q),
        "ci_width": float(high_q - low_q),
        "n_bootstrap": int(n_bootstrap),
        "includes_zero": bool(low_q <= 0 <= high_q),
    }


def cost_sensitivity(
    panel: pd.DataFrame,
    signal_col: str,
    cost_levels_bps: Sequence[float] = (0, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100),
    n_long: int = 10,
    n_short: int = 0,
    rebalance: str = "ME",
    regime_gate: pd.Series | None = None,
) -> pd.DataFrame:
    """Sweep over transaction-cost levels.

    For each cost level, runs the backtest and reports CAGR, Sharpe and the
    realized annual cost drag (computed correctly from per-rebalance turnover,
    not from a daily average — that would average in the zero days and
    badly under-state the true drag).

    Returns:
        DataFrame indexed implicitly by row order with one row per cost level.
    """
    rows = []
    for cost_bps in cost_levels_bps:
        result = run_backtest(
            panel,
            signal_col=signal_col,
            n_long=n_long,
            n_short=n_short,
            rebalance=rebalance,
            cost_bps_per_side=float(cost_bps),
            regime_gate=regime_gate,
        )
        r = result.daily_returns
        # Realized annual cost drag = sum of per-rebalance costs in the sample,
        # annualized over the actual time elapsed.
        n_years = len(r) / TRADING_DAYS_PER_YEAR
        total_cost = (result.turnover * cost_bps / 1e4).sum()
        annual_cost_drag = total_cost / n_years if n_years > 0 else float("nan")
        rows.append({
            "cost_bps_per_side": float(cost_bps),
            "cagr": float(annualized_return(r)),
            "sharpe": float(sharpe_ratio(r)),
            "n_rebalances": int(len(result.rebalance_dates)),
            "avg_turnover_per_rebal": float(result.turnover.mean()),
            "annual_cost_drag": float(annual_cost_drag),
        })
    return pd.DataFrame(rows)
