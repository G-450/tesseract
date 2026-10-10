"""Evaluation package for CGCNN formation energy prediction."""

from src.evaluation.evaluator import EvaluationResult, Evaluator
from src.evaluation.experiment_report import (
    FIGURE_TITLES,
    ExperimentReporter,
    ExplainabilitySummary,
    PredictionSummary,
    TrainingSummary,
    summarize_explanations,
    summarize_predictions,
    summarize_training_log,
)
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_bias,
    compute_mae,
    compute_r2,
    compute_rmse,
)
from src.evaluation.reporter import (
    DEFAULT_SUMMARY_ARTIFACT_LINKS,
    PUBLISHED_CGCNN_BASELINE_MAE,
    Reporter,
)

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "Reporter",
    "compute_mae",
    "compute_rmse",
    "compute_r2",
    "compute_bias",
    "compute_all_metrics",
    "PUBLISHED_CGCNN_BASELINE_MAE",
    "DEFAULT_SUMMARY_ARTIFACT_LINKS",
    "ExperimentReporter",
    "TrainingSummary",
    "ExplainabilitySummary",
    "PredictionSummary",
    "summarize_training_log",
    "summarize_explanations",
    "summarize_predictions",
    "FIGURE_TITLES",
]
