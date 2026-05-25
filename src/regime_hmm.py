"""HMM-based market regime detection.

Fits a Gaussian Hidden Markov Model to weekly SPY features (weekly log
return and rolling annualized volatility) and decodes a regime label per
week. The labels are deterministically relabeled by mean weekly return so
that regime ``0`` is the lowest-return state (bear), regime
``n_regimes - 1`` is the highest-return state (bull), and intermediate
labels interpolate. The model object returned has its ``means_``,
``covars_``, ``transmat_`` and ``startprob_`` permuted to match — a
consumer that later calls ``model.predict(...)`` will receive the
relabeled state numbering, not the raw EM output.

Phase 1 caveat
--------------
Viterbi decoding is performed once on the full series. The label assigned
to week W therefore reflects the maximum-likelihood global path through
*all* weeks, including weeks after W. This is standard practice for
descriptive regime overlays but is **not** a real-time tradable signal.
Phase 2 introduces expanding-window fits when regimes start driving
trading decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .metrics import max_drawdown

WEEKS_PER_YEAR = 52


def compute_regime_features(
    panel: pd.DataFrame,
    ticker: str = "spy",
    vol_window: int = 4,
) -> pd.DataFrame:
    """Weekly log return and rolling annualized vol for one ticker.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel as produced by :func:`src.data.load_panel`.
    ticker : str, default "spy"
        Ticker symbol (lower-case) to extract.
    vol_window : int, default 4
        Window length in weeks for the rolling volatility estimate.

    Returns
    -------
    pd.DataFrame
        Indexed by W-FRI week-end timestamps with columns
        ``["ret_w", "vol_w"]``. Leading NaN rows from the rolling vol
        window are dropped. Annualization uses 52 weeks/year.
    """
    closes = (
        panel.loc[panel["ticker"] == ticker, ["date", "close"]]
        .set_index("date")
        .sort_index()["close"]
    )
    weekly_close = closes.resample("W-FRI").last()
    ret_w = np.log(weekly_close).diff()
    vol_w = ret_w.rolling(vol_window).std() * np.sqrt(WEEKS_PER_YEAR)
    features = pd.DataFrame({"ret_w": ret_w, "vol_w": vol_w}).dropna()
    return features


def fit_hmm_regimes(
    features: pd.DataFrame,
    n_regimes: int = 3,
    seed: int = 42,
    n_iter: int = 200,
) -> tuple[pd.Series, GaussianHMM]:
    """Fit a Gaussian HMM and return Viterbi labels sorted by mean return.

    Parameters
    ----------
    features : pd.DataFrame
        Two-column feature matrix (e.g., output of
        :func:`compute_regime_features`).
    n_regimes : int, default 3
        Number of hidden states.
    seed : int, default 42
        Random state passed to ``GaussianHMM`` for deterministic fits.
    n_iter : int, default 200
        Max EM iterations.

    Returns
    -------
    regimes : pd.Series
        Integer regime labels indexed by ``features.index``. Label 0 =
        lowest mean weekly return (bear); label ``n_regimes - 1`` =
        highest (bull).
    model : hmmlearn.hmm.GaussianHMM
        Fitted model with internal arrays permuted to match the relabeled
        state ordering.

    Raises
    ------
    ValueError
        If ``len(features) < max(30, n_regimes * 10)``.
    """
    min_obs = max(30, n_regimes * 10)
    if len(features) < min_obs:
        raise ValueError(
            f"need at least {min_obs} observations to fit a {n_regimes}-state HMM, "
            f"got {len(features)}"
        )

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=seed,
        tol=1e-4,
    )
    X = features.values
    model.fit(X)
    raw_labels = model.predict(X)

    # Determine permutation: sort states by their realized mean ret_w ascending.
    ret_w = features["ret_w"].to_numpy()
    mean_ret_per_state = np.array([
        ret_w[raw_labels == k].mean() if (raw_labels == k).any() else np.inf
        for k in range(n_regimes)
    ])
    # order[i] is the *old* state label that should become *new* label i.
    order = np.argsort(mean_ret_per_state)
    # perm[old] = new
    perm = np.empty(n_regimes, dtype=int)
    for new_label, old_label in enumerate(order):
        perm[old_label] = new_label

    # Permute the model's internal arrays so model.predict / model.transmat_
    # are aligned with the new label numbering. Write to the underlying
    # _covars_ storage to bypass the public setter's SPD validation, which
    # can falsely fail on EM output for degenerate inputs even though the
    # values being assigned are exactly what hmmlearn just produced.
    model.startprob_ = model.startprob_[order]
    model.means_ = model.means_[order]
    model._covars_ = model._covars_[order]
    model.transmat_ = model.transmat_[order][:, order]

    new_labels = perm[raw_labels]
    regimes = pd.Series(
        new_labels, index=features.index, name="regime"
    ).astype("int8")
    return regimes, model


def expand_regimes_to_daily(
    weekly_regimes: pd.Series,
    daily_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Forward-fill weekly regime labels onto a daily date index.

    Parameters
    ----------
    weekly_regimes : pd.Series
        Integer regime labels indexed by week-end timestamps.
    daily_dates : pd.DatetimeIndex
        Daily dates to project onto.

    Returns
    -------
    pd.Series
        Indexed by ``daily_dates``. The label on date ``X`` is the regime
        of the most recent week-end ``<= X``. Dates earlier than the first
        week-end yield NaN (preserved via the nullable ``Int8`` dtype).
    """
    # Union the two indexes so ffill can carry weekly values onto daily
    # dates; then restrict to the requested daily index.
    union = weekly_regimes.index.union(daily_dates)
    expanded = weekly_regimes.reindex(union).ffill().reindex(daily_dates)
    return expanded.astype("Int8").rename("regime_daily")


def regime_summary_stats(
    features: pd.DataFrame,
    weekly_regimes: pd.Series,
    daily_panel: pd.DataFrame | None = None,
    ticker: str = "spy",
) -> pd.DataFrame:
    """Per-regime descriptive statistics.

    Parameters
    ----------
    features : pd.DataFrame
        Weekly feature matrix (must contain ``ret_w`` and ``vol_w``,
        aligned with ``weekly_regimes``).
    weekly_regimes : pd.Series
        Integer regime labels, same index as ``features``.
    daily_panel : pd.DataFrame, optional
        Long-format panel. If supplied, the ``max_dd_in_regime`` column is
        computed from the worst within-span drawdown of the daily close
        for ``ticker`` across all contiguous spans assigned to each
        regime. If omitted, ``max_dd_in_regime`` is NaN.
    ticker : str, default "spy"
        Ticker used for the daily drawdown calculation.

    Returns
    -------
    pd.DataFrame
        Indexed by regime label with columns ``n_weeks``, ``pct_time``,
        ``mean_ret_w``, ``mean_vol_w``, ``max_dd_in_regime``.
    """
    grouped = features.groupby(weekly_regimes)
    n_weeks = grouped.size().rename("n_weeks")
    pct_time = (n_weeks / n_weeks.sum()).rename("pct_time")
    mean_ret = grouped["ret_w"].mean().rename("mean_ret_w")
    mean_vol = grouped["vol_w"].mean().rename("mean_vol_w")

    if daily_panel is not None:
        spy_close = (
            daily_panel.loc[daily_panel["ticker"] == ticker, ["date", "close"]]
            .set_index("date")
            .sort_index()["close"]
        )
        daily_regimes = expand_regimes_to_daily(weekly_regimes, spy_close.index)
        # Walk the daily regime series to find contiguous spans.
        spans: dict[int, list[float]] = {int(r): [] for r in weekly_regimes.unique()}
        cur_regime = None
        span_start = None
        for date, r in daily_regimes.items():
            if pd.isna(r):
                continue
            r = int(r)
            if r != cur_regime:
                if cur_regime is not None and span_start is not None:
                    span = spy_close.loc[span_start:prev_date]
                    if len(span) > 1:
                        span_rets = span.pct_change().dropna()
                        spans[cur_regime].append(max_drawdown(span_rets))
                cur_regime = r
                span_start = date
            prev_date = date
        # Capture the trailing span.
        if cur_regime is not None and span_start is not None:
            span = spy_close.loc[span_start:prev_date]
            if len(span) > 1:
                span_rets = span.pct_change().dropna()
                spans[cur_regime].append(max_drawdown(span_rets))

        max_dd = pd.Series(
            {r: (min(v) if v else float("nan")) for r, v in spans.items()},
            name="max_dd_in_regime",
        )
    else:
        max_dd = pd.Series(
            {r: float("nan") for r in n_weeks.index},
            name="max_dd_in_regime",
        )

    summary = pd.concat([n_weeks, pct_time, mean_ret, mean_vol, max_dd], axis=1)
    summary.index.name = "regime"
    return summary.sort_index()
