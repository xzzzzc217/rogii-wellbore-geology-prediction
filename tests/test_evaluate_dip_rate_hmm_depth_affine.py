from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_dip_rate_hmm_depth_affine.py"


def _module():
    spec = importlib.util.spec_from_file_location("depth_affine", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_fit_recovers_interior_solution() -> None:
    module = _module()
    result = module._bounded_two_coefficient_fit(
        np.eye(2, dtype=float), np.asarray([1.25, 2.0], dtype=float)
    )
    np.testing.assert_allclose(result, [1.25, 2.0], atol=1e-10)


def test_bounded_fit_clips_to_box_optimum() -> None:
    module = _module()
    result = module._bounded_two_coefficient_fit(
        np.eye(2, dtype=float), np.asarray([-4.0, 9.0], dtype=float), upper=2.3
    )
    np.testing.assert_allclose(result, [0.0, 2.3], atol=1e-10)


def test_depth_basis_is_endpoint_linear() -> None:
    module = _module()
    correction = np.ones(5, dtype=float)
    early, late = module._depth_basis(correction)
    np.testing.assert_allclose(early + late, correction)
    assert early[0] == 1.0 and early[-1] == 0.0
    assert late[0] == 0.0 and late[-1] == 1.0
