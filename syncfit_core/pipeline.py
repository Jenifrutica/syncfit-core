"""High-level deterministic pipeline.

Ties the pieces together: the Ring Buffer accumulates the 100 Hz PPG stream, the
DSP layer extracts R-R intervals and RMSSD, the feature layer builds the numeric
vector, the model predicts central fatigue, the load module computes `k_load` and
the Directed State Graph resolves the physiological phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import SAMPLE_RATE_HZ, extract_rr_intervals, rmssd
from .enums import FatigueLevel, InferredPhase, Modality
from .features import FeatureVector, build_feature_vector
from .graph import DirectedStateGraph, default_cycle_graph, infer_phase_from_day
from .load import LoadDecision, evaluate_load
from .models import FatigueModel
from .structures import RingBuffer


@dataclass(frozen=True)
class EngineResult:
    """Complete deterministic evaluation for one assessment."""

    phase_inferred: InferredPhase
    fatigue_probability: float
    fatigue_level: FatigueLevel
    k_load: float
    rmssd_hrv_ms: float
    features: FeatureVector

    def as_dict(self) -> dict[str, object]:
        return {
            "phase_inferred": self.phase_inferred.value,
            "fatigue_probability": self.fatigue_probability,
            "fatigue_level": self.fatigue_level.value,
            "k_load_multiplier": self.k_load,
            "rmssd_hrv_ms": self.rmssd_hrv_ms,
            "features": self.features.as_dict(),
        }


class SyncFitEngine:
    """Stateful engine for one athlete session."""

    def __init__(
        self,
        model: FatigueModel,
        window_size: int = 256,
        fs: float = SAMPLE_RATE_HZ,
        graph: DirectedStateGraph | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._model = model
        self._fs = fs
        self._graph = graph or default_cycle_graph()
        self._ppg = RingBuffer[float](capacity=window_size)

    @property
    def graph(self) -> DirectedStateGraph:
        return self._graph

    def ingest(self, samples: np.ndarray | list[float]) -> None:
        """Append PPG samples to the fixed Ring Buffer."""
        self._ppg.extend(np.asarray(samples, dtype=float).ravel().tolist())

    def reset(self) -> None:
        self._ppg.clear()

    def current_rmssd(self) -> float:
        """RMSSD computed from the current PPG window (0.0 if not enough data)."""
        if self._ppg.size < 8:
            return 0.0
        rr = extract_rr_intervals(self._ppg.to_numpy(), fs=self._fs)
        return rmssd(rr)

    def evaluate(
        self,
        modality: Modality | str,
        day_or_week: int,
        delta_temperature_c: float,
        isometric_force_loss_pct: float,
        rmssd_hrv_ms: float | None = None,
    ) -> EngineResult:
        """Evaluate biomarkers and return the deterministic decision.

        When `rmssd_hrv_ms` is not provided it is derived from the accumulated
        PPG window.
        """
        modality_value = modality if isinstance(modality, Modality) else Modality(modality)
        effective_rmssd = self.current_rmssd() if rmssd_hrv_ms is None else float(rmssd_hrv_ms)

        features = build_feature_vector(
            modality=modality_value,
            day_or_week=day_or_week,
            delta_temperature_c=delta_temperature_c,
            rmssd_hrv_ms=effective_rmssd,
            isometric_force_loss_pct=isometric_force_loss_pct,
        )
        decision: LoadDecision = evaluate_load(features, self._model)
        phase = infer_phase_from_day(modality_value, day_or_week)
        return EngineResult(
            phase_inferred=phase,
            fatigue_probability=decision.fatigue_probability,
            fatigue_level=decision.fatigue_level,
            k_load=decision.k_load,
            rmssd_hrv_ms=effective_rmssd,
            features=features,
        )


def process_signal(signal: np.ndarray, fs: float = SAMPLE_RATE_HZ) -> float:
    """Convenience helper: PPG signal in, RMSSD out."""
    return rmssd(extract_rr_intervals(signal, fs=fs))


__all__ = ["EngineResult", "SyncFitEngine", "process_signal"]
