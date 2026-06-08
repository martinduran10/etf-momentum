"""Phase 1 invariance check.

Future phases layer additive overlays on top of Phase 1 (the Stack Portfolio
engine). They must never perturb Phase 1's combined daily-return series. This
test pins the Phase 1 headline metrics to a byte-for-byte snapshot frozen at
the moment Phase 2 was layered on; any drift here means a downstream phase
(or some other change) has altered Phase 1, which is forbidden.
"""

from __future__ import annotations

from src.data import load_closes
from src.metrics import summary
from src.stack_backtest import run_stack_portfolio

PHASE1_FROZEN = {
    "total_return": 1.1098811890014075,
    "annualized_return": 0.10634603027694095,
    "annualized_volatility": 0.16206370037402767,
    "sharpe_ratio": 0.6561989515943693,
    "max_drawdown": -0.2854151802877341,
}


def test_phase1_metrics_unchanged() -> None:
    """Phase 1 headline metrics must be byte-for-byte identical to baseline."""
    result = run_stack_portfolio(load_closes())
    actual = summary(result["headline_returns"])
    for name, expected in PHASE1_FROZEN.items():
        assert actual[name] == expected, (
            f"{name}: {actual[name]!r} drifted from frozen {expected!r}"
        )
