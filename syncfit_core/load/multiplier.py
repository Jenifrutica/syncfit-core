"""Deterministic load multiplier (`k_load`) computation.

The multiplier is a pure function of the predicted central-fatigue probability.
It is never produced by a generative model, guaranteeing reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..enums import FatigueLevel
from ..features import FeatureVector
from ..models import FatigueModel

K_LOAD_MIN = 0.70
K_LOAD_MAX = 1.05
K_LOAD_SPAN = K_LOAD_MAX - K_LOAD_MIN

LOW_THRESHOLD = 0.34
HIGH_THRESHOLD = 0.67


def fatigue_level_from_probability(probability: float) -> FatigueLevel:
    """Map a fatigue probability in [0, 1] to a discrete level."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be within [0, 1]")
    if probability < LOW_THRESHOLD:
        return FatigueLevel.LOW
    if probability < HIGH_THRESHOLD:
        return FatigueLevel.MEDIUM
    return FatigueLevel.HIGH


def compute_k_load(fatigue_probability: float) -> float:
    """Return `k_load` in [0.70, 1.05] from the fatigue probability.

    A probability of 0 yields the maximum allowed load (1.05); a probability of
    1 yields the minimum (0.70).
    """
    if not 0.0 <= fatigue_probability <= 1.0:
        raise ValueError("fatigue_probability must be within [0, 1]")
    value = K_LOAD_MAX - K_LOAD_SPAN * fatigue_probability
    return float(np.clip(value, K_LOAD_MIN, K_LOAD_MAX))


@dataclass(frozen=True)
class LoadDecision:
    """Result of the deterministic evaluation for one telemetry sample."""

    fatigue_probability: float
    fatigue_level: FatigueLevel
    k_load: float
    features: FeatureVector

    def as_dict(self) -> dict[str, object]:
        return {
            "fatigue_probability": self.fatigue_probability,
            "fatigue_level": self.fatigue_level.value,
            "k_load_multiplier": self.k_load,
            "features": self.features.as_dict(),
        }


def evaluate_load(features: FeatureVector, model: FatigueModel) -> LoadDecision:
    """Predict fatigue and compute the load decision for one feature vector."""
    probability = float(model.predict_fatigue_probability(features.to_array())[0])
    return LoadDecision(
        fatigue_probability=probability,
        fatigue_level=fatigue_level_from_probability(probability),
        k_load=compute_k_load(probability),
        features=features,
    )


__all__ = [
    "K_LOAD_MIN",
    "K_LOAD_MAX",
    "LOW_THRESHOLD",
    "HIGH_THRESHOLD",
    "fatigue_level_from_probability",
    "compute_k_load",
    "LoadDecision",
    "evaluate_load",
]
