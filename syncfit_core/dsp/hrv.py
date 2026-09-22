"""Heart-rate variability metrics."""

from __future__ import annotations

import numpy as np


def rmssd(rr_ms: np.ndarray) -> float:
    """Root mean square of successive R-R differences, in milliseconds.

    Returns 0.0 when fewer than two intervals are available.
    """
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 2:
        return 0.0
    successive = np.diff(rr)
    return float(np.sqrt(np.mean(successive**2)))


def mean_hr_bpm(rr_ms: np.ndarray) -> float:
    """Mean heart rate in beats per minute derived from R-R intervals."""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size == 0:
        return 0.0
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0:
        return 0.0
    return 60_000.0 / mean_rr


__all__ = ["rmssd", "mean_hr_bpm"]
