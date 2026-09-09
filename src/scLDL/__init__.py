"""scLDL: single-cell label distribution learning and annotation."""

from scLDL.benchmark import ALL_METHODS, CORE_METHODS, run_benchmark, summarize
from scLDL.metrics import classification_metrics, distribution_metrics, score_annotations
from scLDL.models import (
    ANNOTATION_MODELS,
    ConcentrationLE,
    HybridLEVI,
    ImprovedLEVI,
    LEVI,
    LIBLE,
    MLPBaseline,
    StateConcentrationLE,
    InterpretableLE,
)
from scLDL.pipeline import AnnotationPipeline

__all__ = [
    "AnnotationPipeline",
    "MLPBaseline",
    "LIBLE",
    "LEVI",
    "ImprovedLEVI",
    "ConcentrationLE",
    "StateConcentrationLE",
    "InterpretableLE",
    "HybridLEVI",
    "ANNOTATION_MODELS",
    "classification_metrics",
    "distribution_metrics",
    "score_annotations",
    "run_benchmark",
    "summarize",
    "CORE_METHODS",
    "ALL_METHODS",
]
