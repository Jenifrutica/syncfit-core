import numpy as np
import pytest

from syncfit_core.contracts_adapter import (
    features_from_telemetry_frame,
    result_to_adapted_routine,
)
from syncfit_core.enums import Modality
from syncfit_core.features import build_feature_vector
from syncfit_core.load import LoadDecision, evaluate_load
from syncfit_core.models import train_default_model
from syncfit_core.pipeline import EngineResult


FRAME = {
    "schema_version": "1.0.0",
    "device_id": "esp32-syncfit-01",
    "session_id": "3f1b2c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
    "timestamp": "2026-09-21T13:24:05Z",
    "modality": "MENSTRUAL_CYCLE",
    "day_or_week": 14,
    "biomarkers": {
        "delta_temperature_c": 0.42,
        "rmssd_hrv_ms": 28.5,
        "isometric_force_loss_pct": 12.8,
    },
}


def test_features_from_telemetry_frame():
    kwargs = features_from_telemetry_frame(FRAME)
    fv = build_feature_vector(**kwargs)
    assert fv.day_or_week == 14
    assert fv.rmssd_hrv_ms == 28.5


def test_result_to_adapted_routine_shape():
    model = train_default_model(n_samples=800, seed=4)
    features = build_feature_vector(Modality.MENSTRUAL_CYCLE, 14, 0.42, 28.5, 12.8)
    decision: LoadDecision = evaluate_load(features, model)
    result = EngineResult(
        phase_inferred=__import__("syncfit_core").InferredPhase.OVULATORY,
        fatigue_probability=decision.fatigue_probability,
        fatigue_level=decision.fatigue_level,
        k_load=decision.k_load,
        rmssd_hrv_ms=28.5,
        features=features,
    )
    payload = result_to_adapted_routine(result, session_id=FRAME["session_id"])
    assert payload["phase_inferred"] == "OVULATORY"
    assert 0.70 <= payload["k_load_multiplier"] <= 1.05
    assert payload["adapted_routine"] == []


def test_validate_adapted_routine_requires_contracts():
    syncfit_contracts = pytest.importorskip("syncfit_contracts")
    from syncfit_core.contracts_adapter import validate_adapted_routine

    payload = {
        "schema_version": "1.0.0",
        "session_id": "3f1b2c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
        "phase_inferred": "LUTEAL",
        "fatigue_level": "MEDIUM",
        "k_load_multiplier": 0.9,
        "alerts": [],
        "adapted_routine": [],
    }
    model = validate_adapted_routine(payload)
    assert model.k_load_multiplier == 0.9
    assert syncfit_contracts is not None
