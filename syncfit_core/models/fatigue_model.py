"""Supervised tabular fatigue classifier.

Random Forest by default, with optional XGBoost when installed. The model
outputs the probability of central fatigue, which the load module turns into the
deterministic `k_load` multiplier.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]


class ModelBackend(str, Enum):
    RANDOM_FOREST = "RANDOM_FOREST"
    XGBOOST = "XGBOOST"


def _build_estimator(backend: ModelBackend, random_state: int) -> Any:
    if backend is ModelBackend.XGBOOST:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "XGBoost backend requested but xgboost is not installed. "
                "Install it with `pip install xgboost` or use ModelBackend.RANDOM_FOREST."
            ) from exc
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )


class FatigueModel:
    """Wraps a fitted classifier that predicts central-fatigue probability."""

    def __init__(
        self,
        backend: ModelBackend = ModelBackend.RANDOM_FOREST,
        random_state: int = 42,
    ) -> None:
        self.backend = backend
        self.random_state = random_state
        self._estimator = _build_estimator(backend, random_state)
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "FatigueModel":
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2:
            raise ValueError("features must be a 2-D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and labels must have the same length")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on an empty dataset")
        self._estimator.fit(x, y)
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("model is not fitted; call fit() or load() first")

    def predict_fatigue_probability(self, features: np.ndarray) -> np.ndarray:
        """Return the probability of central fatigue for each row."""
        self._check_fitted()
        x = np.atleast_2d(np.asarray(features, dtype=float))
        probabilities = self._estimator.predict_proba(x)
        classes = list(self._estimator.classes_)
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        return np.zeros(x.shape[0], dtype=float)

    def save(self, path: str | Path) -> Path:
        """Persist the fitted model with joblib."""
        if joblib is None:  # pragma: no cover
            raise ImportError("joblib is required to persist models")
        self._check_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "FatigueModel":
        if joblib is None:  # pragma: no cover
            raise ImportError("joblib is required to load models")
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError("file does not contain a FatigueModel")
        return loaded


__all__ = ["FatigueModel", "ModelBackend"]
