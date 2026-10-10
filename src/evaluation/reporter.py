"""Reporter module for exporting evaluation metrics, predictions, and summaries."""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.evaluation.evaluator import EvaluationResult

PUBLISHED_CGCNN_BASELINE_MAE = 0.039  # eV/atom on matbench_mp_e_form

DEFAULT_SUMMARY_ARTIFACT_LINKS: Dict[str, str] = {
    "Metrics JSON": "metrics.json",
    "Sample Predictions CSV": "predictions.csv",
    "Parity Plot": "parity_plot.png",
    "Residual Distribution Plot": "residual_plot.png",
    "Prediction Histogram": "prediction_histogram.png",
    "Training Loss Curves": "loss_curve.png",
    "Learning Rate Schedule": "learning_rate.png",
}
"""Artifact links used when the caller does not supply its own layout."""


class Reporter:
    """Exports evaluation artifacts
    including metrics.json, predictions.csv, and summary.md."""

    def __init__(self, baseline_mae: float = PUBLISHED_CGCNN_BASELINE_MAE) -> None:
        """Initializes the Reporter.

        Args:
            baseline_mae: Published benchmark baseline MAE in eV/atom.
        """
        self.baseline_mae = baseline_mae

    def export(
        self,
        result: EvaluationResult,
        output_dir: str,
        artifact_links: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Exports all evaluation reports to the given output directory.

        Args:
            result: The EvaluationResult containing predictions and metrics.
            output_dir: Target directory path for exported files.
            artifact_links: Optional mapping of artifact label to a path,
                relative to ``output_dir``, listed at the end of the summary.
                Defaults to the flat layout where every artifact sits beside
                the summary.

        Returns:
            Dict[str, str]: Dictionary mapping artifact names to their file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        metrics_path = os.path.join(output_dir, "metrics.json")
        predictions_path = os.path.join(output_dir, "predictions.csv")
        summary_path = os.path.join(output_dir, "summary.md")

        self.export_metrics_json(result, metrics_path)
        self.export_predictions_csv(result, predictions_path)
        self.export_summary_md(result, summary_path, artifact_links=artifact_links)

        return {
            "metrics_json": metrics_path,
            "predictions_csv": predictions_path,
            "summary_md": summary_path,
        }

    def export_metrics_json(self, result: EvaluationResult, output_path: str) -> None:
        """Exports numeric metrics and benchmark comparison to JSON.

        Args:
            result: The EvaluationResult object.
            output_path: File path to save metrics.json.
        """
        mae = result.metrics.get("mae", 0.0)
        mae_diff = mae - self.baseline_mae
        pct_change = (
            (mae_diff / self.baseline_mae) * 100.0 if self.baseline_mae > 0 else 0.0
        )

        metrics_payload: Dict[str, Any] = {
            "num_samples": len(result.predictions),
            "mae": float(mae),
            "rmse": float(result.metrics.get("rmse", 0.0)),
            "r2": float(result.metrics.get("r2", 0.0)),
            "bias": float(result.metrics.get("bias", 0.0)),
            "benchmark_comparison": {
                "benchmark_dataset": "matbench_mp_e_form",
                "baseline_name": "Published CGCNN Baseline",
                "baseline_mae_ev_atom": self.baseline_mae,
                "current_mae_ev_atom": float(mae),
                "mae_difference_ev_atom": float(mae_diff),
                "relative_difference_pct": float(pct_change),
                "status": "better" if mae <= self.baseline_mae else "higher_error",
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=4)

    def export_predictions_csv(
        self, result: EvaluationResult, output_path: str
    ) -> None:
        """Exports per-sample predictions, targets, residuals, and errors to CSV.

        Args:
            result: The EvaluationResult object.
            output_path: File path to save predictions.csv.
        """
        df = pd.DataFrame(
            {
                "sample_id": result.sample_ids,
                "true_target": result.targets.numpy(),
                "predicted_target": result.predictions.numpy(),
                "residual": result.residuals.numpy(),
                "absolute_error": result.absolute_errors.numpy(),
            }
        )
        df.to_csv(output_path, index=False)

    def export_summary_md(
        self,
        result: EvaluationResult,
        output_path: str,
        artifact_links: Optional[Dict[str, str]] = None,
    ) -> None:
        """Exports a formatted Markdown evaluation report.

        Args:
            result: The EvaluationResult object.
            output_path: File path to save summary.md.
            artifact_links: Optional mapping of artifact label to a path
                relative to the summary, listed under "Exported Artifacts".
                Defaults to the flat layout where every artifact sits beside
                the summary.
        """
        num_samples = len(result.predictions)
        mae = result.metrics.get("mae", 0.0)
        rmse = result.metrics.get("rmse", 0.0)
        r2 = result.metrics.get("r2", 0.0)
        bias = result.metrics.get("bias", 0.0)

        abs_errors = result.absolute_errors.numpy()
        residuals = result.residuals.numpy()
        targets = result.targets.numpy()
        predictions = result.predictions.numpy()

        median_ae = float(np.median(abs_errors)) if len(abs_errors) > 0 else 0.0
        std_res = float(np.std(residuals)) if len(residuals) > 0 else 0.0
        p95_ae = float(np.percentile(abs_errors, 95)) if len(abs_errors) > 0 else 0.0
        max_ae = float(np.max(abs_errors)) if len(abs_errors) > 0 else 0.0
        min_ae = float(np.min(abs_errors)) if len(abs_errors) > 0 else 0.0

        mae_diff = mae - self.baseline_mae
        pct_change = (
            (mae_diff / self.baseline_mae) * 100.0 if self.baseline_mae > 0 else 0.0
        )

        # Sort indices for top best and top worst predictions
        sorted_indices = np.argsort(abs_errors)
        top_k = min(5, num_samples)
        best_indices = sorted_indices[:top_k]
        worst_indices = sorted_indices[-top_k:][::-1]

        best_rows = []
        for idx in best_indices:
            sid = result.sample_ids[idx]
            row = (
                f"| `{sid}` "
                f"| {targets[idx]:.4f} "
                f"| {predictions[idx]:.4f} "
                f"| {residuals[idx]:+.4f} "
                f"| {abs_errors[idx]:.4f} |"
            )
            best_rows.append(row)

        worst_rows = []
        for idx in worst_indices:
            sid = result.sample_ids[idx]
            row = (
                f"| `{sid}` "
                f"| {targets[idx]:.4f} "
                f"| {predictions[idx]:.4f} "
                f"| {residuals[idx]:+.4f} "
                f"| {abs_errors[idx]:.4f} |"
            )
            worst_rows.append(row)

        comparison_discussion = (
            f"The evaluated model achieves a test Mean Absolute Error of "
            f"**{mae:.4f} eV/atom**, compared to the published CGCNN baseline "
            f"of **{self.baseline_mae:.4f} eV/atom** on `matbench_mp_e_form`. "
        )
        if mae <= self.baseline_mae:
            comparison_discussion += (
                f"This matches or exceeds standard published benchmarks "
                f"by {abs(pct_change):.2f}%."
            )
        else:
            comparison_discussion += (
                f"The difference of {mae_diff:+.4f} eV/atom "
                f"({pct_change:+.2f}%) reflects the current "
                f"training budget, hyperparameter choices, and split partition."
            )
        pred_table_header = (
            "| Sample ID | True Value (eV/atom) | Predicted (eV/atom) | "
            "Residual (eV/atom) | Abs Error (eV/atom) |\n"
            "| :--- | :--- | :--- | :--- | :--- |"
        )
        mean_bias_row = (
            f"| **Mean Bias** | **{bias:+.4f}** | eV/atom | "
            f"Mean signed prediction error ($\\hat{{y}} - y$) |"
        )

        links = (
            artifact_links
            if artifact_links is not None
            else DEFAULT_SUMMARY_ARTIFACT_LINKS
        )
        artifact_lines = "\n".join(
            f"- **{label}**: [`{path}`]({path})" for label, path in links.items()
        )

        markdown_content = f"""# CGCNN Model Evaluation Summary

## Overview
- **Evaluation Dataset**: Matbench Formation Energy (`matbench_mp_e_form`)
- **Total Test Samples**: {num_samples}
- **Primary Metric**: Mean Absolute Error (MAE) in eV/atom

---

## Performance Metrics

| Metric | Value | Unit | Description |
| :--- | :--- | :--- | :--- |
| **MAE** | **{mae:.4f}** | eV/atom | Mean Absolute Error (primary benchmark) |
| **RMSE** | **{rmse:.4f}** | eV/atom | Root Mean Square Error |
| **R²** | **{r2:.4f}** | - | Coefficient of Determination |
{mean_bias_row}

---

## Benchmark Comparison

| Model / Benchmark | MAE (eV/atom) | Difference (eV/atom) | Relative (%) |
| :--- | :--- | :--- | :--- |
| **Published CGCNN Baseline** | {self.baseline_mae:.4f} | - | - |
| **Current Model (Test Set)** | **{mae:.4f}** | {mae_diff:+.4f} | {pct_change:+.2f}% |

### Discussion
{comparison_discussion}

---

## Error Distribution & Analysis

| Statistic | Value (eV/atom) |
| :--- | :--- |
| **Median Absolute Error** | {median_ae:.4f} |
| **Residual Standard Deviation** | {std_res:.4f} |
| **95th Percentile Absolute Error** | {p95_ae:.4f} |
| **Minimum Absolute Error** | {min_ae:.4f} |
| **Maximum Absolute Error** | {max_ae:.4f} |

### Top {top_k} Best Predictions (Lowest Absolute Error)
{pred_table_header}
{chr(10).join(best_rows)}

### Top {top_k} Worst Predictions (Highest Absolute Error)
{pred_table_header}
{chr(10).join(worst_rows)}

---

## Exported Artifacts
{artifact_lines}
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
