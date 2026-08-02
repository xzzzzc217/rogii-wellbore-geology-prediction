from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_translated_twin_transfer import (  # noqa: E402
    _fingerprint_horizontal,
    _fingerprint_typewell,
)


def test_horizontal_fingerprint_ignores_spatial_and_tvt_datum_translation() -> None:
    first = pd.DataFrame(
        {
            "MD": [100.0, 101.0, 102.0],
            "X": [10.0, 11.0, 13.0],
            "Y": [20.0, 20.5, 21.5],
            "Z": [-1.0, -1.2, -1.7],
            "GR": [5.0, np.nan, 7.0],
            "TVT_input": [1000.0, 1000.5, np.nan],
        }
    )
    translated = first.copy()
    translated["MD"] += 500.0
    translated["X"] += 10_000.0
    translated["Y"] -= 20_000.0
    translated["Z"] += 50.0
    translated["TVT_input"] += 123.0
    assert _fingerprint_horizontal(first) == _fingerprint_horizontal(translated)


def test_horizontal_fingerprint_rejects_shape_or_gr_change() -> None:
    first = pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, -1.0, -2.0],
            "GR": [1.0, 2.0, 3.0],
            "TVT_input": [10.0, 11.0, np.nan],
        }
    )
    changed = first.copy()
    changed.loc[1, "GR"] = 2.1
    assert _fingerprint_horizontal(first) != _fingerprint_horizontal(changed)


def test_typewell_fingerprint_ignores_tvt_translation_only() -> None:
    first = pd.DataFrame({"TVT": [100.0, 101.0], "GR": [5.0, 6.0]})
    translated = first.copy()
    translated["TVT"] += 50.0
    assert _fingerprint_typewell(first) == _fingerprint_typewell(translated)
    translated.loc[1, "GR"] += 0.1
    assert _fingerprint_typewell(first) != _fingerprint_typewell(translated)
