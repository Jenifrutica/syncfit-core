import numpy as np
import pytest

from syncfit_core.dsp import (
    bandpass_filter,
    butter_bandpass,
    detect_peaks,
    extract_rr_intervals,
    peaks_to_rr,
    reject_artifacts,
    rmssd,
)


def synthetic_ppg(fs=100, seconds=12, hr_bpm=72, noise=0.02, seed=0):
    t = np.arange(int(fs * seconds)) / fs
    freq = hr_bpm / 60.0
    signal = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t)
    rng = np.random.default_rng(seed)
    return signal + rng.normal(0, noise, signal.size)


def test_butter_bandpass_shapes():
    sos = butter_bandpass()
    assert sos.ndim == 2 and sos.shape[0] == 4


def test_bandpass_rejects_invalid_cutoffs():
    with pytest.raises(ValueError):
        butter_bandpass(lowcut=0.0)
    with pytest.raises(ValueError):
        butter_bandpass(lowcut=4.0, highcut=2.0)
    with pytest.raises(ValueError):
        butter_bandpass(highcut=60.0, fs=100.0)


def test_bandpass_filter_removes_dc():
    fs = 100
    t = np.arange(fs * 10) / fs
    signal = 5.0 + np.sin(2 * np.pi * 1.2 * t)
    filtered = bandpass_filter(signal, fs=fs)
    assert abs(np.mean(filtered)) < 0.5


def test_detect_peaks_count():
    signal = synthetic_ppg(seconds=12, hr_bpm=72)
    peaks = detect_peaks(signal)
    assert 12 <= peaks.size <= 18


def test_peaks_to_rr_and_rmssd():
    peaks = np.array([0, 100, 200, 300])  # 1 s spacing at 100 Hz -> 1000 ms
    rr = peaks_to_rr(peaks, fs=100)
    assert np.allclose(rr, [1000.0, 1000.0, 1000.0])
    assert rmssd(rr) == pytest.approx(0.0)


def test_rmssd_known_value():
    rr = np.array([800.0, 850.0, 800.0, 850.0])
    assert rmssd(rr) == pytest.approx(50.0)


def test_rmssd_insufficient_data():
    assert rmssd(np.array([800.0])) == 0.0


def test_reject_artifacts_removes_outliers():
    rr = np.array([800.0, 810.0, 795.0, 5000.0, 805.0])
    cleaned = reject_artifacts(rr)
    assert 5000.0 not in cleaned
    assert cleaned.size == 4


def test_extract_rr_intervals_from_signal():
    signal = synthetic_ppg(seconds=15, hr_bpm=60, noise=0.01)
    rr = extract_rr_intervals(signal)
    assert rr.size >= 8
    assert np.all(rr >= 300) and np.all(rr <= 2000)
