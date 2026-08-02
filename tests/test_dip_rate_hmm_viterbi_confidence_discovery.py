from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_dip_rate_hmm_viterbi_confidence_discovery.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hmm_viterbi_confidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fusion_is_finite_and_bounded_between_paths() -> None:
    module = _load_module()
    forward = np.asarray([0.0, 1.0, 2.0])
    viterbi = np.asarray([2.0, 0.0, 2.0])
    uncertainty = np.asarray([0.5, 2.0, 1.0])
    fused, weight = module.fuse_paths(forward, uncertainty, viterbi)
    assert np.isfinite(fused).all()
    assert np.all((0.0 <= weight) & (weight <= 1.0))
    assert np.all(fused >= np.minimum(forward, viterbi))
    assert np.all(fused <= np.maximum(forward, viterbi))
    assert fused[2] == forward[2]


def test_more_uncertainty_increases_viterbi_weight_for_same_disagreement() -> None:
    module = _load_module()
    forward = np.zeros(2)
    viterbi = np.ones(2)
    _, weight = module.fuse_paths(
        forward,
        np.asarray([0.5, 2.0]),
        viterbi,
    )
    assert weight[1] > weight[0]
    assert module.OVERLAY_SCALE == 0.40
