"""Tests for the HMM-based regime detection module."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_panel
from src.regime_hmm import (
    compute_regime_features,
    fit_hmm_regimes,
    expand_regimes_to_daily,
    regime_summary_stats,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return load_panel()


@pytest.fixture(scope="module")
def features(panel) -> pd.DataFrame:
    return compute_regime_features(panel)


@pytest.fixture(scope="module")
def regimes_and_model(features):
    return fit_hmm_regimes(features, n_regimes=3, seed=42)


def test_features_shape_and_no_nans(features):
    assert list(features.columns) == ["ret_w", "vol_w"]
    assert not features.isna().any().any()
    assert len(features) >= 500


def test_features_weekly_frequency(features):
    diffs = features.index.to_series().diff().dropna()
    # All consecutive gaps are exactly 7 days (W-FRI resample is uniform).
    assert (diffs == pd.Timedelta(days=7)).all()


def test_fit_is_deterministic(features):
    reg1, model1 = fit_hmm_regimes(features, n_regimes=3, seed=42)
    reg2, model2 = fit_hmm_regimes(features, n_regimes=3, seed=42)
    assert reg1.equals(reg2)
    assert np.allclose(model1.transmat_, model2.transmat_)
    assert np.allclose(model1.means_, model2.means_)


def test_regime_ordering_invariant(features, regimes_and_model):
    regimes, _ = regimes_and_model
    means = features.groupby(regimes)["ret_w"].mean()
    assert means.is_monotonic_increasing, (
        f"regime ordering invariant violated: {means.to_dict()}"
    )


def test_transition_matrix_rows_sum_to_one(regimes_and_model):
    _, model = regimes_and_model
    assert np.allclose(model.transmat_.sum(axis=1), 1.0, atol=1e-8)


def test_startprob_sums_to_one(regimes_and_model):
    _, model = regimes_and_model
    assert np.isclose(model.startprob_.sum(), 1.0, atol=1e-8)


def test_expand_to_daily_no_lookahead(panel, regimes_and_model):
    regimes, _ = regimes_and_model
    daily_dates = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    daily = expand_regimes_to_daily(regimes, daily_dates)
    assert len(daily) == len(daily_dates)

    # Sample 20 random in-range daily dates and verify each one's label is
    # the regime of the latest week-end <= that date.
    rng = np.random.default_rng(0)
    valid = daily.dropna()
    sample = rng.choice(valid.index, size=20, replace=False)
    for d in sample:
        week_ends_le = regimes.index[regimes.index <= d]
        assert len(week_ends_le) > 0
        expected = int(regimes.loc[week_ends_le[-1]])
        assert int(daily.loc[d]) == expected, (
            f"on {d}, daily label {daily.loc[d]} != latest week-end label {expected}"
        )


def test_expand_to_daily_aligned_length(panel, regimes_and_model):
    regimes, _ = regimes_and_model
    daily_dates = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    daily = expand_regimes_to_daily(regimes, daily_dates)
    assert len(daily) == len(daily_dates)
    assert (daily.index == daily_dates).all()


def test_too_short_input_raises():
    rng = np.random.default_rng(0)
    short = pd.DataFrame(
        {"ret_w": rng.normal(0, 0.02, size=5),
         "vol_w": rng.normal(0.15, 0.05, size=5)},
        index=pd.date_range("2024-01-05", periods=5, freq="W-FRI"),
    )
    with pytest.raises(ValueError, match="at least"):
        fit_hmm_regimes(short, n_regimes=3)


def test_near_constant_returns_does_not_crash():
    # Near-constant low-magnitude returns. Realistic stress case: a long
    # stretch of small noise around zero with no strong regime structure.
    rng = np.random.default_rng(0)
    n = 200
    feat = pd.DataFrame(
        {"ret_w": rng.normal(0, 1e-3, size=n),
         "vol_w": np.full(n, 0.05) + rng.normal(0, 1e-3, size=n)},
        index=pd.date_range("2020-01-03", periods=n, freq="W-FRI"),
    )
    regimes, model = fit_hmm_regimes(feat, n_regimes=2, seed=42)
    assert len(regimes) == n
    assert set(regimes.unique()).issubset({0, 1})


def test_regime_summary_stats_shape_and_values(panel, features, regimes_and_model):
    regimes, _ = regimes_and_model
    summary = regime_summary_stats(features, regimes, daily_panel=panel)
    assert list(summary.columns) == [
        "n_weeks", "pct_time", "mean_ret_w", "mean_vol_w", "max_dd_in_regime",
    ]
    assert summary["n_weeks"].sum() == len(features)
    assert np.isclose(summary["pct_time"].sum(), 1.0)
    # Bear regime (label 0) should have the worst drawdown.
    assert summary["max_dd_in_regime"].idxmin() == 0
