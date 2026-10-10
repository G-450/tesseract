"""Inference subsystem for CGCNN formation energy prediction.

Predicts formation energy (eV/atom) for crystal structures outside the
training dataset, from CIF files or pymatgen ``Structure`` objects. Graph
construction reuses the training ``CrystalGraphBuilder`` and ``GraphConfig``
so that inference preprocessing is identical to training preprocessing
(PRD 90). Inference never retrains or modifies the model (PRD 89).

Public API
----------
Predictor
    Loads a frozen checkpoint and predicts single or batched structures.
PredictionResult
    Dataclass holding one structure's prediction and provenance metadata.
BatchPredictionResult
    Dataclass holding a batch run's predictions, failures and summary.
PredictionExporter
    Exports predictions to CSV and JSON.
InferenceError
    Root exception for this subsystem.
"""

from src.inference.exceptions import (
    InferenceError,
    ModelLoadError,
    PredictionError,
    PredictionExportError,
    StructureLoadError,
)
from src.inference.exporter import PredictionExporter
from src.inference.predictor import (
    BatchPredictionResult,
    PredictionResult,
    Predictor,
)

__all__ = [
    "Predictor",
    "PredictionResult",
    "BatchPredictionResult",
    "PredictionExporter",
    "InferenceError",
    "ModelLoadError",
    "StructureLoadError",
    "PredictionError",
    "PredictionExportError",
]
