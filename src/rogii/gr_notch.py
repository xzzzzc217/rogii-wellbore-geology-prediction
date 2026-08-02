"""Target-free GR notch filtering for trajectory experiments.

The transform removes only the strongest non-DC Fourier bin and its immediate
neighbors.  It accepts no target, identifier, leaderboard, or formation input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GRNotchDiagnostics:
    """Audit record for one well's deterministic GR transform."""

    rows: int
    finite_rows: int
    applied: bool
    reason: str
    peak_bin: int | None
    removed_bins: tuple[int, ...]
    rms_move: float
    max_abs_move: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


def notch_strongest_periodic_component(
    gr: np.ndarray,
    *,
    min_finite: int = 64,
    notch_radius: int = 1,
) -> tuple[np.ndarray, GRNotchDiagnostics]:
    """Remove the strongest non-DC periodic GR component.

    Missing input positions remain missing.  Finite gaps are interpolated only
    to calculate the Fourier spectrum, and the inverse transform is copied back
    only at originally finite positions.
    """

    values = np.asarray(gr, dtype=float)
    if values.ndim != 1:
        raise ValueError("gr must be a one-dimensional array")
    if min_finite < 2:
        raise ValueError("min_finite must be at least two")
    if notch_radius < 0:
        raise ValueError("notch_radius cannot be negative")

    output = values.copy()
    finite = np.isfinite(values)
    finite_rows = int(finite.sum())

    def no_op(reason: str) -> tuple[np.ndarray, GRNotchDiagnostics]:
        return output, GRNotchDiagnostics(
            rows=int(values.size),
            finite_rows=finite_rows,
            applied=False,
            reason=reason,
            peak_bin=None,
            removed_bins=(),
            rms_move=0.0,
            max_abs_move=0.0,
        )

    if finite_rows < min_finite:
        return no_op("too_few_finite_rows")

    index = np.arange(values.size, dtype=float)
    filled = values.copy()
    filled[~finite] = np.interp(index[~finite], index[finite], values[finite])
    center = float(np.median(filled))
    spectrum = np.fft.rfft(filled - center)
    if spectrum.size < 5:
        return no_op("spectrum_too_short")

    amplitude = np.abs(spectrum)
    amplitude[0] = 0.0
    peak = int(np.argmax(amplitude))
    if peak <= 0 or not np.isfinite(amplitude[peak]) or amplitude[peak] <= 0.0:
        return no_op("no_nonzero_periodic_peak")

    first = max(1, peak - notch_radius)
    stop = min(spectrum.size, peak + notch_radius + 1)
    removed = tuple(range(first, stop))
    spectrum[list(removed)] = 0.0
    filtered = np.fft.irfft(spectrum, n=values.size) + center
    output[finite] = filtered[finite]
    if not np.all(np.isfinite(output[finite])):
        raise ValueError("GR notch filter produced non-finite values")

    move = output[finite] - values[finite]
    return output, GRNotchDiagnostics(
        rows=int(values.size),
        finite_rows=finite_rows,
        applied=True,
        reason="applied",
        peak_bin=peak,
        removed_bins=removed,
        rms_move=float(np.sqrt(np.mean(move**2))),
        max_abs_move=float(np.max(np.abs(move))),
    )
