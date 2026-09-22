"""Supervised model and training pipeline."""

from .fatigue_model import FatigueModel, ModelBackend
from .training import make_synthetic_dataset, train_default_model

__all__ = [
    "FatigueModel",
    "ModelBackend",
    "make_synthetic_dataset",
    "train_default_model",
]
