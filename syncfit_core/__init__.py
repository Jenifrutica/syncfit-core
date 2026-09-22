"""SyncFit Core - deterministic numerical engine of SyncFit Edge.

Digital signal processing, the trained tabular fatigue model and the
deterministic `k_load` load multiplier. This is the on-premise, trained half of
the artificial intelligence; the cloud reasoning half lives in
`syncfit-ai-reasoning`.
"""

from .enums import CyclePhase, FatigueLevel, InferredPhase, Modality, Trimester
from .features import FEATURE_NAMES, FeatureVector, build_feature_vector
from .graph import (
    DirectedStateGraph,
    StateNode,
    StateTransition,
    default_cycle_graph,
    infer_phase_from_day,
)
from .load import (
    K_LOAD_MAX,
    K_LOAD_MIN,
    LoadDecision,
    compute_k_load,
    evaluate_load,
    fatigue_level_from_probability,
)
from .models import FatigueModel, ModelBackend, make_synthetic_dataset, train_default_model
from .pipeline import EngineResult, SyncFitEngine, process_signal
from .structures import FenwickTree, RingBuffer, SlidingWindow

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # enums
    "Modality",
    "CyclePhase",
    "Trimester",
    "InferredPhase",
    "FatigueLevel",
    # structures
    "RingBuffer",
    "FenwickTree",
    "SlidingWindow",
    "DirectedStateGraph",
    "StateNode",
    "StateTransition",
    "default_cycle_graph",
    "infer_phase_from_day",
    # features
    "FeatureVector",
    "build_feature_vector",
    "FEATURE_NAMES",
    # load
    "K_LOAD_MIN",
    "K_LOAD_MAX",
    "LoadDecision",
    "compute_k_load",
    "evaluate_load",
    "fatigue_level_from_probability",
    # models
    "FatigueModel",
    "ModelBackend",
    "make_synthetic_dataset",
    "train_default_model",
    # pipeline
    "EngineResult",
    "SyncFitEngine",
    "process_signal",
]
