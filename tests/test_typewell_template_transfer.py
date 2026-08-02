from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_typewell_template_transfer import _interpolate_template  # noqa: E402


def test_interpolate_template_preserves_endpoints() -> None:
    result = _interpolate_template(np.asarray([0.0, 2.0, 4.0]), 5)
    assert np.allclose(result, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_singleton_template_is_constant() -> None:
    result = _interpolate_template(np.asarray([3.5]), 4)
    assert np.allclose(result, 3.5)
