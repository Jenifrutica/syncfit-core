import numpy as np
import pytest

from syncfit_core.enums import FatigueLevel, Modality
from syncfit_core.features import build_feature_vector
from syncfit_core.load import (
    K_LOAD_MAX,
    K_LOAD_MIN,
    compute_k_load,
    evaluate_load,
    fatigue_level_from_probability,
)
from syncfit_core.models import train_default_model


def test_compute_k_load_bounds():
    assert compute_k_load(0.0) == pytest.approx(K_LOAD_MAX)
    assert compute_k_load(1.0) == pytest.approx(K_LOAD_MIN)
    assert K_LOAD_MIN <= compute_k_load(0.5) <= K_LOAD_MAX


def test_compute_k_load_monotonic_decreasing():
    values = [compute_k_load(p) for p in np.linspace(0, 1, 11)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_compute_k_load_invalid():
    with pytest.raises(ValueError):
        compute_k_load(1.5)


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, FatigueLevel.LOW),
        (0.33, FatigueLevel.LOW),
        (0.5, FatigueLevel.MEDIUM),
        (0.66, FatigueLevel.MEDIUM),
        (0.9, FatigueLevel.HIGH),
    ],
)
def test_fatigue_level_thresholds(probability, expected):
    assert fatigue_level_from_probability(probability) is expected


def test_evaluate_load_end_to_end():
    model = train_default_model(n_samples=1500, seed=1)
    features = build_feature_vector(Modality.MENSTRUAL_CYCLE, 15, 0.4, 25.0, 30.0)
    decision = evaluate_load(features, model)
    assert 0.0 <= decision.fatigue_probability <= 1.0
    assert K_LOAD_MIN <= decision.k_load <= K_LOAD_MAX
    assert decision.as_dict()["fatigue_level"] in {"LOW", "MEDIUM", "HIGH"}
