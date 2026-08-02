from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_dip_rate_hmm_viterbi_discovery.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hmm_viterbi_discovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_viterbi_kernel_returns_finite_global_path() -> None:
    module = _load_module()
    steps = 8
    grid = np.arange(-3.0, 3.5, 0.5)
    rates = np.linspace(-0.1, 0.1, module.RATE_STATES)
    emission = np.full((steps, len(grid)), -8.0, dtype=np.float64)
    emission[:4, np.argmin(np.abs(grid + 1.0))] = 0.0
    emission[4:, np.argmin(np.abs(grid - 1.0))] = 0.0
    delta_md = np.ones(steps, dtype=np.float64)
    delta_z = np.zeros(steps, dtype=np.float64)
    path = module._dip_rate_viterbi(
        emission,
        delta_md,
        delta_z,
        grid,
        rates,
        -1.0,
        0.0,
    )
    assert path.shape == (steps,)
    assert np.isfinite(path).all()
    assert np.all(np.isin(path, grid))


def test_viterbi_path_respects_one_step_state_dynamics() -> None:
    module = _load_module()
    grid = np.arange(-2.0, 2.5, 0.5)
    rates = np.linspace(-0.1, 0.1, module.RATE_STATES)
    emission = np.zeros((6, len(grid)), dtype=np.float64)
    path = module._dip_rate_viterbi(
        emission,
        np.ones(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        grid,
        rates,
        0.0,
        0.0,
    )
    assert np.max(np.abs(np.diff(path))) <= 1.0
