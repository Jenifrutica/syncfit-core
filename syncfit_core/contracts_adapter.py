"""Adapters between the deterministic core and `syncfit-contracts`.

The core stays dependency-free; these helpers convert to and from the contract
models only when `syncfit-contracts` is installed. They are thin and import the
package lazily so the numerical engine can run on its own.
"""

from __future__ import annotations

from typing import Any

from .pipeline import EngineResult


def _require_contracts() -> Any:
    try:
        import syncfit_contracts  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "syncfit-contracts is not installed. Install it to use the contract "
            "adapters (see https://github.com/Jenifrutica/syncfit-contracts)."
        ) from exc
    return syncfit_contracts


def features_from_telemetry_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Extract the keyword arguments for `build_feature_vector` from a frame.

    Accepts a raw telemetry-frame dictionary (as emitted by the device) so the
    core does not need to import the contracts package to consume it.
    """
    biomarkers = frame["biomarkers"]
    return {
        "modality": frame["modality"],
        "day_or_week": frame["day_or_week"],
        "delta_temperature_c": biomarkers["delta_temperature_c"],
        "rmssd_hrv_ms": biomarkers["rmssd_hrv_ms"],
        "isometric_force_loss_pct": biomarkers["isometric_force_loss_pct"],
    }


def result_to_adapted_routine(
    result: EngineResult,
    session_id: str,
    adapted_routine: list[dict[str, Any]] | None = None,
    alerts: list[str] | None = None,
) -> dict[str, Any]:
    """Build an `AdaptedRoutine`-shaped payload from an engine result.

    The reasoning layer is responsible for the exercise-level adaptations; the
    core contributes the phase, fatigue level and the deterministic `k_load`.
    """
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "session_id": session_id,
        "phase_inferred": result.phase_inferred.value,
        "fatigue_level": result.fatigue_level.value,
        "k_load_multiplier": result.k_load,
        "alerts": alerts or [],
        "adapted_routine": adapted_routine or [],
    }
    return payload


def validate_adapted_routine(payload: dict[str, Any]) -> Any:
    """Validate a payload against the contracts model, if installed."""
    contracts = _require_contracts()
    return contracts.AdaptedRoutine.model_validate(payload)


__all__ = [
    "features_from_telemetry_frame",
    "result_to_adapted_routine",
    "validate_adapted_routine",
]
