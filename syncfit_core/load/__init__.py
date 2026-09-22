"""Deterministic load multiplier."""

from .multiplier import (
    HIGH_THRESHOLD,
    K_LOAD_MAX,
    K_LOAD_MIN,
    LOW_THRESHOLD,
    LoadDecision,
    compute_k_load,
    evaluate_load,
    fatigue_level_from_probability,
)

__all__ = [
    "HIGH_THRESHOLD",
    "K_LOAD_MAX",
    "K_LOAD_MIN",
    "LOW_THRESHOLD",
    "LoadDecision",
    "compute_k_load",
    "evaluate_load",
    "fatigue_level_from_probability",
]
