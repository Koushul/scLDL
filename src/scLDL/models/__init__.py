from scLDL.models.concentration import ConcentrationLE
from scLDL.models.concentration_state import StateConcentrationLE
from scLDL.models.hybrid import HybridLEVI
from scLDL.models.improved_levi import ImprovedLEVI
from scLDL.models.interpretable import InterpretableLE
from scLDL.models.levi import LEVI
from scLDL.models.lible import LIBLE
from scLDL.models.mlp import MLPBaseline

ANNOTATION_MODELS = {
    "mlp": MLPBaseline,
    "lible": LIBLE,
    "concentration": ConcentrationLE,
    "state_concentration": StateConcentrationLE,
    "interpretable": InterpretableLE,
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
    "StateConcentrationLE",
    "InterpretableLE",
    "HybridLEVI",
    "ANNOTATION_MODELS",
    "ENHANCEMENT_MODELS",
]
