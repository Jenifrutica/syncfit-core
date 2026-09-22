"""Feature vector construction for the supervised tabular model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..enums import Modality

MODALITY_ENCODING: dict[Modality, int] = {
    Modality.MENSTRUAL_CYCLE: 0,
    Modality.GESTATIONAL: 1,
}

FEATURE_NAMES: tuple[str, ...] = (
    "modality",
    "day_or_week",
    "delta_temperature_c",
    "rmssd_hrv_ms",
    "isometric_force_loss_pct",
)


@dataclass(frozen=True)
class FeatureVector:
    """Numeric vector ingested by the fatigue classifier."""

    modality: Modality
    day_or_week: int
    delta_temperature_c: float
    rmssd_hrv_ms: float
    isometric_force_loss_pct: float

    def to_array(self) -> np.ndarray:
        return np.asarray(
            [
                float(MODALITY_ENCODING[self.modality]),
                float(self.day_or_week),
                float(self.delta_temperature_c),
                float(self.rmssd_hrv_ms),
                float(self.isometric_force_loss_pct),
            ],
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.to_array().tolist()))


def build_feature_vector(
    modality: Modality | str,
    day_or_week: int,
    delta_temperature_c: float,
    rmssd_hrv_ms: float,
    isometric_force_loss_pct: float,
) -> FeatureVector:
    """Validate and build a `FeatureVector` from raw biomarker readings."""
    modality_value = modality if isinstance(modality, Modality) else Modality(modality)
    if not 1 <= day_or_week <= 42:
        raise ValueError("day_or_week must be within [1, 42]")
    if not -2.0 <= delta_temperature_c <= 2.0:
        raise ValueError("delta_temperature_c must be within [-2.0, 2.0]")
    if not 0.0 <= rmssd_hrv_ms <= 300.0:
        raise ValueError("rmssd_hrv_ms must be within [0, 300]")
    if not 0.0 <= isometric_force_loss_pct <= 100.0:
        raise ValueError("isometric_force_loss_pct must be within [0, 100]")
    return FeatureVector(
        modality=modality_value,
        day_or_week=int(day_or_week),
        delta_temperature_c=float(delta_temperature_c),
        rmssd_hrv_ms=float(rmssd_hrv_ms),
        isometric_force_loss_pct=float(isometric_force_loss_pct),
    )


__all__ = ["FeatureVector", "build_feature_vector", "FEATURE_NAMES", "MODALITY_ENCODING"]
