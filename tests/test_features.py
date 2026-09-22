import pytest

from syncfit_core.enums import Modality
from syncfit_core.features import FEATURE_NAMES, build_feature_vector


def test_build_feature_vector_and_array():
    fv = build_feature_vector(
        modality=Modality.MENSTRUAL_CYCLE,
        day_or_week=14,
        delta_temperature_c=0.42,
        rmssd_hrv_ms=28.5,
        isometric_force_loss_pct=12.8,
    )
    assert fv.to_array().shape == (len(FEATURE_NAMES),)
    assert fv.as_dict()["modality"] == 0.0
    assert fv.as_dict()["day_or_week"] == 14.0


def test_gestational_modality_encoding():
    fv = build_feature_vector("GESTATIONAL", 18, 0.1, 34.0, 9.0)
    assert fv.as_dict()["modality"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"day_or_week": 0},
        {"day_or_week": 43},
        {"delta_temperature_c": 3.0},
        {"rmssd_hrv_ms": -1.0},
        {"isometric_force_loss_pct": 101.0},
    ],
)
def test_validation_errors(kwargs):
    base = dict(
        modality=Modality.MENSTRUAL_CYCLE,
        day_or_week=14,
        delta_temperature_c=0.2,
        rmssd_hrv_ms=30.0,
        isometric_force_loss_pct=10.0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        build_feature_vector(**base)
