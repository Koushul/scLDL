from scLDL.models.concentration import ConcentrationLE
from scLDL.models.hybrid import HybridLEVI
from scLDL.models.improved_levi import ImprovedLEVI
from scLDL.models.levi import LEVI
from scLDL.models.lible import LIBLE
from scLDL.models.mlp import MLPBaseline

ANNOTATION_MODELS = {
    "mlp": MLPBaseline,
    "lible": LIBLE,
    "concentration": ConcentrationLE,
    "hybrid": HybridLEVI,
}

ENHANCEMENT_MODELS = {
    "levi": LEVI,
    "improved_levi": ImprovedLEVI,
}

__all__ = [
    "MLPBaseline",
    "LIBLE",
    "LEVI",
    "ImprovedLEVI",
    "ConcentrationLE",
    "HybridLEVI",
    "ANNOTATION_MODELS",
    "ENHANCEMENT_MODELS",
]
