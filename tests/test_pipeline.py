import numpy as np

from syncfit_core import SyncFitEngine, train_default_model
from syncfit_core.enums import InferredPhase, Modality
from syncfit_core.pipeline import process_signal


def synthetic_ppg(fs=100, seconds=12, hr_bpm=70, seed=0):
    t = np.arange(int(fs * seconds)) / fs
    freq = hr_bpm / 60.0
    rng = np.random.default_rng(seed)
    signal = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t)
    return signal + rng.normal(0, 0.01, signal.size)


def test_engine_evaluate_with_explicit_rmssd():
    engine = SyncFitEngine(train_default_model(n_samples=1000, seed=2), window_size=256)
    result = engine.evaluate(
        modality=Modality.MENSTRUAL_CYCLE,
        day_or_week=15,
        delta_temperature_c=0.42,
        isometric_force_loss_pct=12.8,
        rmssd_hrv_ms=28.5,
    )
    assert result.phase_inferred is InferredPhase.OVULATORY
    assert 0.70 <= result.k_load <= 1.05
    assert result.as_dict()["rmssd_hrv_ms"] == 28.5


def test_engine_derives_rmssd_from_ppg_window():
    engine = SyncFitEngine(train_default_model(n_samples=1000, seed=2), window_size=1024)
    engine.ingest(synthetic_ppg(seconds=12, hr_bpm=60))
    result = engine.evaluate(
        modality="MENSTRUAL_CYCLE",
        day_or_week=3,
        delta_temperature_c=0.1,
        isometric_force_loss_pct=5.0,
    )
    assert result.phase_inferred is InferredPhase.MENSTRUAL
    assert result.rmssd_hrv_ms > 0.0


def test_engine_reset():
    engine = SyncFitEngine(train_default_model(n_samples=500, seed=2), window_size=128)
    engine.ingest(np.ones(128))
    engine.reset()
    assert engine.current_rmssd() == 0.0


def test_process_signal_helper():
    value = process_signal(synthetic_ppg(seconds=12, hr_bpm=66))
    assert value >= 0.0
