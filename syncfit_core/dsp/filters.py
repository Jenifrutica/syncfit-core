"""Digital signal processing primitives.

Butterworth IIR band-pass filtering of the optical PPG signal, exactly as the
technical document specifies (0.5 Hz - 5.0 Hz).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

LOWCUT_HZ = 0.5
HIGHCUT_HZ = 5.0
SAMPLE_RATE_HZ = 100.0
FILTER_ORDER = 4


def butter_bandpass(
    lowcut: float = LOWCUT_HZ,
    highcut: float = HIGHCUT_HZ,
    fs: float = SAMPLE_RATE_HZ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Design a Butterworth band-pass filter in second-order sections form."""
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("cutoff frequencies must be positive")
    if highcut >= fs / 2:
        raise ValueError("highcut must be below the Nyquist frequency")
    if lowcut >= highcut:
        raise ValueError("lowcut must be below highcut")
    nyquist = 0.5 * fs
    return butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = LOWCUT_HZ,
    highcut: float = HIGHCUT_HZ,
    fs: float = SAMPLE_RATE_HZ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter to `signal`."""
    data = np.asarray(signal, dtype=float)
    if data.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if data.size == 0:
        raise ValueError("signal must not be empty")
    sos = butter_bandpass(lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    padlen = min(3 * (sos.shape[0] * 2), data.size - 1)
    if padlen <= 0:
        return data.copy()
    return sosfiltfilt(sos, data, padlen=padlen)


__all__ = ["butter_bandpass", "bandpass_filter", "LOWCUT_HZ", "HIGHCUT_HZ", "SAMPLE_RATE_HZ"]
