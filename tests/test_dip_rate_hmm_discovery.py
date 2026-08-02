from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_dip_rate_hmm_discovery import (  # noqa: E402
    DISAGREEMENT_CLIP,
    DISCOVERY_WELLS,
    GRID_STEP,
    OVERLAY_SCALE,
    RATE_STATES,
    predict_hidden_hmm,
)


def test_discovery_contract_is_frozen() -> None:
    assert DISCOVERY_WELLS == 32
    assert GRID_STEP == 0.50
    assert RATE_STATES == 21
    assert OVERLAY_SCALE == 0.20
    assert DISAGREEMENT_CLIP == 5.0


def test_dip_rate_hmm_returns_finite_hidden_track() -> None:
    rows = 80
    hidden_start = 45
    md = np.arange(rows, dtype=float)
    z = 1000.0 - 0.05 * md
    true_tvt = 1200.0 + 0.02 * md
    gr = 80.0 + 12.0 * np.sin(true_tvt / 4.0)
    tvt_input = true_tvt.copy()
    tvt_input[hidden_start:] = np.nan
    horizontal = pd.DataFrame(
        {"MD": md, "Z": z, "GR": gr, "TVT_input": tvt_input}
    )
    tw_tvt = np.linspace(1170.0, 1230.0, 600)
    typewell = pd.DataFrame(
        {"TVT": tw_tvt, "GR": 80.0 + 12.0 * np.sin(tw_tvt / 4.0)}
    )
    prediction, uncertainty = predict_hidden_hmm(horizontal, typewell)
    assert prediction.shape == (rows - hidden_start,)
    assert uncertainty.shape == prediction.shape
    assert np.isfinite(prediction).all()
    assert np.isfinite(uncertainty).all()
