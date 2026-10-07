"""Explainability subsystem for CGCNN formation energy predictions.

Provides attribution methods (Integrated Gradients, GNNExplainer) that
identify which atoms and bonds most influenced a given prediction.

Note:
    All outputs are model-level interpretations, not physical causal
    evidence (PRD §82, §101).

Public API
----------
BaseExplainer
    Abstract base class for all explainability methods.
ExplanationResult
    Dataclass holding attribution scores and summary statistics.
IntegratedGradientsExplainer
    Captum-based attribution via Integrated Gradients.
GNNExplainerWrapper
    PyG GNNExplainer with graceful fallback.
Exporter
    Batch exporter for JSON/PNG/CSV artifacts.
ExplainabilityError
    Root exception for this subsystem.
"""

from src.explainability.base import BaseExplainer, ExplanationResult
from src.explainability.exceptions import (
    ExplainabilityError,
    ExplainerInitializationError,
    ExportError,
    SampleExplanationError,
)
from src.explainability.exporter import Exporter
from src.explainability.gnn_explainer import GNNExplainerWrapper
from src.explainability.integrated_gradients import IntegratedGradientsExplainer

__all__ = [
    "BaseExplainer",
    "ExplanationResult",
    "IntegratedGradientsExplainer",
    "GNNExplainerWrapper",
    "Exporter",
    "ExplainabilityError",
    "ExplainerInitializationError",
    "SampleExplanationError",
    "ExportError",
]
