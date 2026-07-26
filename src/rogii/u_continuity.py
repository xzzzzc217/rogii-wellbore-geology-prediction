"""Target-free boundary continuity in ``U = TVT + Z``.

This module is a small, testable transcription of the late Guard640
post-processing layer.  It deliberately accepts no target/ground-truth array.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UContinuityDiagnostics:
    """Per-well diagnostics for the frozen boundary correction."""

    rows: int
    last_visible_row: int
    first_hidden_row: int
    boundary_gap_u_before: float
    boundary_gap_u_after: float
    cap: float
    tau: float
    first_move: float
    mean_abs_move: float
    max_abs_move: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready diagnostics."""

        return asdict(self)


def apply_u_continuity(
    prediction: np.ndarray,
    md: np.ndarray,
    z: np.ndarray,
    tvt_input: np.ndarray,
    *,
    cap: float = 8.0,
    tau: float = 240.0,
) -> tuple[np.ndarray, UContinuityDiagnostics]:
    """Apply the frozen Guard640 boundary fade to a contiguous hidden suffix.

    The exact correction is

    ``move = -clip(first_hidden_U - last_visible_U, -cap, cap)
    * exp(-(hidden_MD - last_visible_MD) / tau)``.

    Only rows where ``tvt_input`` is missing are changed.  No true TVT target is
    accepted or consulted.
    """

    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("cap must be finite and positive")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")

    values = {
        "prediction": np.asarray(prediction, dtype=float),
        "md": np.asarray(md, dtype=float),
        "z": np.asarray(z, dtype=float),
        "tvt_input": np.asarray(tvt_input, dtype=float),
    }
    lengths = {name: value.size for name, value in values.items() if value.ndim == 1}
    if len(lengths) != len(values) or len(set(lengths.values())) != 1:
        raise ValueError("prediction, md, z, and tvt_input must be equal-length 1D arrays")
    if not np.all(np.isfinite(values["prediction"])):
        raise ValueError("prediction contains non-finite values")
    if not np.all(np.isfinite(values["md"])) or not np.all(np.isfinite(values["z"])):
        raise ValueError("md and z must be finite")

    visible = np.flatnonzero(np.isfinite(values["tvt_input"]))
    hidden = np.flatnonzero(~np.isfinite(values["tvt_input"]))
    if visible.size == 0:
        raise ValueError("continuity requires a visible TVT_input prefix")
    if hidden.size == 0:
        raise ValueError("continuity requires a hidden suffix")
    if hidden.size > 1 and not np.array_equal(hidden[1:], hidden[:-1] + 1):
        raise ValueError("continuity requires a contiguous hidden suffix")

    last_visible = int(visible[-1])
    first_hidden = int(hidden[0])
    if first_hidden != last_visible + 1 or int(hidden[-1]) != values["prediction"].size - 1:
        raise ValueError("continuity hidden suffix must immediately follow the visible prefix")

    md_since = values["md"][hidden] - values["md"][last_visible]
    if not np.all(np.isfinite(md_since)) or np.any(md_since <= 0.0):
        raise ValueError("hidden measured depth must be after the last visible row")

    refined = values["prediction"].copy()
    last_visible_u = float(values["tvt_input"][last_visible] + values["z"][last_visible])
    gap_u = float(values["prediction"][first_hidden] + values["z"][first_hidden] - last_visible_u)
    move = -np.clip(gap_u, -cap, cap) * np.exp(-md_since / tau)
    refined[hidden] = values["prediction"][hidden] + move
    if not np.all(np.isfinite(refined)):
        raise ValueError("continuity produced non-finite predictions")
    if float(np.max(np.abs(move))) > cap + 1e-10:
        raise ValueError("continuity move exceeds its cap")

    diagnostics = UContinuityDiagnostics(
        rows=int(hidden.size),
        last_visible_row=last_visible,
        first_hidden_row=first_hidden,
        boundary_gap_u_before=gap_u,
        boundary_gap_u_after=float(
            refined[first_hidden] + values["z"][first_hidden] - last_visible_u
        ),
        cap=float(cap),
        tau=float(tau),
        first_move=float(move[0]),
        mean_abs_move=float(np.mean(np.abs(move))),
        max_abs_move=float(np.max(np.abs(move))),
    )
    return refined, diagnostics
