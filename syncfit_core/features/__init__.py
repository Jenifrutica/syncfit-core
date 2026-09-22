"""Feature engineering for the deterministic core."""

from .extractor import (
    FEATURE_NAMES,
    MODALITY_ENCODING,
    FeatureVector,
    build_feature_vector,
)

__all__ = [
    "FEATURE_NAMES",
    "MODALITY_ENCODING",
    "FeatureVector",
    "build_feature_vector",
]
