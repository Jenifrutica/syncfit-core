"""Domain enumerations for the deterministic core.

These mirror the values defined in `syncfit-contracts` so the numerical engine
stays dependency-free while remaining wire-compatible with the rest of the
system.
"""

from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """Physiological modality selected by the athlete."""

    MENSTRUAL_CYCLE = "MENSTRUAL_CYCLE"
    GESTATIONAL = "GESTATIONAL"


class CyclePhase(str, Enum):
    MENSTRUAL = "MENSTRUAL"
    FOLLICULAR = "FOLLICULAR"
    OVULATORY = "OVULATORY"
    LUTEAL = "LUTEAL"


class Trimester(str, Enum):
    TRIMESTER_1 = "TRIMESTER_1"
    TRIMESTER_2 = "TRIMESTER_2"
    TRIMESTER_3 = "TRIMESTER_3"


class InferredPhase(str, Enum):
    MENSTRUAL = "MENSTRUAL"
    FOLLICULAR = "FOLLICULAR"
    OVULATORY = "OVULATORY"
    LUTEAL = "LUTEAL"
    TRIMESTER_1 = "TRIMESTER_1"
    TRIMESTER_2 = "TRIMESTER_2"
    TRIMESTER_3 = "TRIMESTER_3"


class FatigueLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


__all__ = [
    "Modality",
    "CyclePhase",
    "Trimester",
    "InferredPhase",
    "FatigueLevel",
]
