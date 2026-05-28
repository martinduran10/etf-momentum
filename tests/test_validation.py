"""CI GATE — reproduce the Excel Stack Portfolio headline metrics.

Each metric must land within +/-3% (relative) of its Excel target. The build
fails if the reproduction drifts outside tolerance.
"""

from __future__ import annotations

import pytest

from src.data import load_closes
from src.metrics import summary
from src.stack_backtest import run_stack_portfolio

# Excel backtest targets (headline window 2015-12-07 -> end of data).
TARGETS = {
    "total_return": 1.0858,
    "annualized_return": 0.1082,
    "annualized_volatility": 0.1616,
    "sharpe_ratio": 0.67,
    "max_drawdown": -0.2849,
}

# Relative tolerance per metric.
REL_TOL = 0.03


@pytest.fixture(scope="module")
def metrics() -> dict[str, float]:
    result = run_stack_portfolio(load_closes())
    return summary(result["headline_returns"])


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_metric_within_relative_tolerance(
    metrics: dict[str, float], name: str
) -> None:
    target = TARGETS[name]
    actual = metrics[name]
    rel = abs(actual - target) / abs(target)
    assert rel <= REL_TOL, (
        f"{name}: {actual:.4f} is {rel:.1%} off target {target:.4f} "
        f"(tolerance {REL_TOL:.0%})"
    )


def test_sharpe_in_explicit_band(metrics: dict[str, float]) -> None:
    # Spec calls out the Sharpe band explicitly: 0.65 .. 0.69.
    assert 0.65 <= metrics["sharpe_ratio"] <= 0.69
