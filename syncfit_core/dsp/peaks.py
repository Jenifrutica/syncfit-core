"""R-R peak detection and artifact rejection over the filtered PPG signal."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .filters import SAMPLE_RATE_HZ, bandpass_filter

MIN_RR_S = 0.30
MAX_RR_S = 2.00
MAX_DEVIATION = 0.30


def detect_peaks(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE_HZ,
    min_rr_s: float = MIN_RR_S,
    max_rr_s: float = MAX_RR_S,
) -> np.ndarray:
    """Detect pulse peaks and return their sample indices."""
    data = np.asarray(signal, dtype=float)
    if data.size < 3:
        return np.empty(0, dtype=int)
    min_distance = max(1, int(round(min_rr_s * fs)))
    prominence = max(0.05 * (np.max(data) - np.min(data)), 1e-9)
    peaks, _ = find_peaks(data, distance=min_distance, prominence=prominence)
    return peaks.astype(int)


def peaks_to_rr(peaks: np.ndarray, fs: float = SAMPLE_RATE_HZ) -> np.ndarray:
    """Convert peak sample indices to R-R intervals in milliseconds."""
    if peaks.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(peaks) / fs * 1000.0


def reject_artifacts(
    rr_ms: np.ndarray,
    min_rr_ms: float = MIN_RR_S * 1000.0,
    max_rr_ms: float = MAX_RR_S * 1000.0,
    max_deviation: float = MAX_DEVIATION,
) -> np.ndarray:
    """Remove physiologically implausible R-R intervals.

    Intervals outside the absolute bounds are dropped. When at least three
    intervals remain, those deviating more than `max_deviation` from the median
    are also dropped.
    """
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[(rr >= min_rr_ms) & (rr <= max_rr_ms)]
    if rr.size >= 3:
        median = float(np.median(rr))
        if median > 0:
            deviation = np.abs(rr - median) / median
            rr = rr[deviation <= max_deviation]
    return rr


def extract_rr_intervals(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE_HZ,
    filter_signal: bool = True,
) -> np.ndarray:
    """Full pipeline: optional band-pass filter, peak detection, artifact rejection."""
    data = np.asarray(signal, dtype=float)
    if filter_signal and data.size > 8:
        data = bandpass_filter(data, fs=fs)
    peaks = detect_peaks(data, fs=fs)
    rr = peaks_to_rr(peaks, fs=fs)
    return reject_artifacts(rr)


__all__ = [
    "detect_peaks",
    "peaks_to_rr",
    "reject_artifacts",
    "extract_rr_intervals",
    "MIN_RR_S",
    "MAX_RR_S",
]
