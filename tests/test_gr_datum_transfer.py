from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_gr_datum_transfer import (  # noqa: E402
    MATERIAL_DELTA,
    _fit_scale,
    _gate,
)


def test_weighted_no_intercept_scale_fit_and_guard() -> None:
    raw = np.asarray([1.0, 2.0, 3.0])
    targets = 0.10 * raw
    rows = np.asarray([100, 200, 300])
    assert np.isclose(_fit_scale(raw, targets, rows), 0.10)
    assert np.isclose(_fit_scale(raw, 10.0 * raw, rows), 0.25)
    assert np.isclose(_fit_scale(np.zeros(3), targets, rows), 0.0)


def test_gate_requires_material_stable_improvement() -> None:
    passing = {
        "fold_count": 7,
        "pooled_delta": MATERIAL_DELTA - 0.001,
        "p95_well_delta": -0.01,
        "worst_well_delta": 0.20,
        "nonpositive_well_fraction": 0.46,
        "improved_folds": 5,
    }
    assert all(_gate(passing, 0.96).values())
    assert not _gate(dict(passing, pooled_delta=-0.01), 0.96)[
        "material_pooled_improvement"
    ]
    assert not _gate(passing, 0.94)["raw_signal_success_fraction"]
