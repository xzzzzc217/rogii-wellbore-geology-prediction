from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_dip_rate_hmm_smoother_discovery.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hmm_smoother_discovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoother_kernel_is_finite_and_uses_future_evidence() -> None:
    module = _load_module()
    steps = 8
    grid = np.arange(-3.0, 3.5, 0.5)
    rates = np.linspace(-0.1, 0.1, module.RATE_STATES)
    emission = np.zeros((steps, len(grid)), dtype=np.float64)
    emission[:, :] = -8.0
    emission[:4, np.argmin(np.abs(grid + 1.0))] = 0.0
    emission[4:, np.argmin(np.abs(grid - 1.0))] = 0.0
    delta_md = np.ones(steps, dtype=np.float64)
    delta_z = np.zeros(steps, dtype=np.float64)
    forward, smoother = module._dip_rate_filter_and_smoother(
        emission,
        delta_md,
        delta_z,
        grid,
        rates,
        -1.0,
        0.0,
    )
    assert forward.shape == (steps,)
    assert smoother.shape == (steps,)
    assert np.isfinite(forward).all()
    assert np.isfinite(smoother).all()
    assert np.isclose(forward[-1], smoother[-1], atol=1e-5)
    assert not np.allclose(forward[:-1], smoother[:-1])


def test_discovery_selection_is_frozen() -> None:
    module = _load_module()
    assert module.DISCOVERY_WELLS == 24
    assert module.OVERLAY_SCALE == 0.4
    assert module.DISAGREEMENT_CLIP == 5.0
    assert len(module.EXPECTED_SELECTED_WELLS) == 24
    assert len(set(module.EXPECTED_SELECTED_WELLS)) == 24
