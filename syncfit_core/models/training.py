"""Training pipeline and synthetic dataset generation.

Real labelled data is not available yet, so this module generates a
domain-informed synthetic dataset (autonomic and neuromuscular fatigue coupled to
phase and thermal load) to train and validate the pipeline end to end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..features import FEATURE_NAMES
from .fatigue_model import FatigueModel, ModelBackend


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_dataset(
    n_samples: int = 4000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic `(features, labels)` dataset.

    The latent fatigue risk couples low HRV, high isometric strength loss,
    elevated thermal delta and late-cycle / late-gestation phase.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)

    modality = rng.integers(0, 2, size=n_samples)  # 0 = menstrual, 1 = gestational
    day_or_week = np.where(
        modality == 0,
        rng.integers(1, 36, size=n_samples),
        rng.integers(1, 43, size=n_samples),
    ).astype(float)
    rmssd = np.clip(rng.normal(45.0, 15.0, size=n_samples), 5.0, 120.0)
    force_loss = np.clip(rng.normal(8.0, 7.0, size=n_samples), 0.0, 60.0)
    delta_temp = np.clip(rng.normal(0.15, 0.25, size=n_samples), -1.0, 1.5)

    cycle_load = np.where(modality == 0, day_or_week / 35.0, day_or_week / 42.0)
    risk = (
        0.45 * (1.0 - rmssd / 120.0)
        + 0.35 * (force_loss / 60.0)
        + 0.10 * np.clip(delta_temp, 0.0, None)
        + 0.10 * cycle_load
    )
    probability = _sigmoid(6.0 * (risk - 0.5))
    labels = (rng.random(n_samples) < probability).astype(int)

    features = np.column_stack([modality, day_or_week, delta_temp, rmssd, force_loss])
    return features, labels


def train_default_model(
    n_samples: int = 4000,
    seed: int = 42,
    backend: ModelBackend = ModelBackend.RANDOM_FOREST,
) -> FatigueModel:
    """Train a `FatigueModel` on the synthetic dataset."""
    features, labels = make_synthetic_dataset(n_samples=n_samples, seed=seed)
    model = FatigueModel(backend=backend, random_state=seed)
    return model.fit(features, labels)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the SyncFit Edge fatigue model.")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=[b.value for b in ModelBackend],
        default=ModelBackend.RANDOM_FOREST.value,
    )
    parser.add_argument("--output", type=Path, default=Path("models/artifacts/fatigue_model.joblib"))
    args = parser.parse_args(argv)

    model = train_default_model(
        n_samples=args.samples,
        seed=args.seed,
        backend=ModelBackend(args.backend),
    )
    path = model.save(args.output)
    print(f"Trained {args.backend} model on {args.samples} samples -> {path}")
    print(f"Feature order: {', '.join(FEATURE_NAMES)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
