from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_dip_rate_hmm_confirmation import (  # noqa: E402
    POOLED_DELTA_LIMIT,
    WELL_WIN_FRACTION_LIMIT,
    WORKERS,
    _discovery_indices,
)


def test_confirmation_contract_is_frozen() -> None:
    assert WORKERS == 8
    assert POOLED_DELTA_LIMIT == -0.05
    assert WELL_WIN_FRACTION_LIMIT == 0.65


def test_discovery_partition_uses_sha_order() -> None:
    rows = []
    for index in range(40):
        rows.append(
            {
                "well": f"well-{index:02d}",
                "folds": {"spatial_cluster_7_seed_20260727": 0},
            }
        )
    metadata = {"per_well": rows}
    selected = _discovery_indices(metadata)
    expected = sorted(
        range(40),
        key=lambda index: hashlib.sha256(
            f"well-{index:02d}".encode()
        ).hexdigest(),
    )[:32]
    assert selected == set(expected)
