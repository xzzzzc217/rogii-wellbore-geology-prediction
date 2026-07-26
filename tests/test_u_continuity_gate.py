from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rogii.u_continuity_gate import (  # noqa: E402
    deployment_decision,
    paired_well_bootstrap,
    pooled_rmse,
)


def test_paired_bootstrap_is_deterministic_and_preserves_improvement() -> None:
    baseline = np.array([10.0, 12.0, 8.0, 14.0, 7.0])
    corrected = baseline - 1.0
    rows = np.array([100, 200, 300, 400, 500])

    first = paired_well_bootstrap(baseline, corrected, rows, replicates=500, seed=42)
    second = paired_well_bootstrap(baseline, corrected, rows, replicates=500, seed=42)

    assert first == second
    assert first["ci95_upper"] < 0.0
    assert pooled_rmse(corrected, rows) < pooled_rmse(baseline, rows)


def test_deployment_gate_requires_every_frozen_condition() -> None:
    passing = deployment_decision(
        pooled_delta=-0.021,
        macro_delta=-0.001,
        bootstrap_ci_upper=-0.0001,
        cluster_deltas=[-0.1, -0.02, 0.0, -0.001, 0.019],
    )
    assert passing["decision"] == "GO"
    assert all(passing["checks"].values())

    failing = deployment_decision(
        pooled_delta=-0.019,
        macro_delta=-0.001,
        bootstrap_ci_upper=-0.0001,
        cluster_deltas=[-0.1, -0.02, 0.0, -0.001, 0.021],
    )
    assert failing["decision"] == "NO-GO"
    assert not failing["checks"]["pooled_delta_at_most_minus_0_020"]
    assert not failing["checks"]["no_spatial_cluster_worse_than_plus_0_020"]
