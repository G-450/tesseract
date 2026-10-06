"""Batch exporter for explainability results.

Writes per-material JSON and PNG files plus an aggregated summary CSV
for an entire batch of ExplanationResult objects (PRD §87, §98).

Note:
    Outputs are model-level interpretations, not physical causal evidence.
"""

import csv
import json
import logging
import os
from typing import List, Optional

from src.explainability.base import ExplanationResult
from src.explainability.exceptions import ExportError

logger = logging.getLogger(__name__)


class Exporter:
    """Utility class that persists a batch of explanation results to disk.

    Produces the directory layout required by the PRD (§87, §98):

    .. code-block:: text

        output_dir/
            summary.csv
            <sample_id>/
                explanation.json
                heatmap.png

    Note:
        Outputs are model-level interpretations, not physical causal evidence
        (PRD §82, §101).
    """

    @staticmethod
    def visualize(
        result: ExplanationResult, output_path: Optional[str] = None
    ) -> Optional[str]:
        """Generate a heatmap figure for node and edge importances."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Sample {result.sample_id} — "
            f"Pred: {result.prediction:.4f}  Target: {result.target:.4f}  "
            f"Method: {result.metadata.get('method', 'N/A')}",
            fontsize=10,
        )

        node_data = result.node_importance.cpu().numpy().reshape(1, -1)
        im0 = axes[0].imshow(node_data, aspect="auto", cmap="Reds")
        axes[0].set_title("Node (Atom) Importance")
        axes[0].set_xlabel("Atom Index")
        axes[0].set_yticks([])
        fig.colorbar(im0, ax=axes[0], label="Attribution Score")

        edge_data = result.edge_importance.cpu().numpy().reshape(1, -1)
        im1 = axes[1].imshow(edge_data, aspect="auto", cmap="Reds")
        axes[1].set_title("Edge (Bond) Importance")
        axes[1].set_xlabel("Edge Index")
        axes[1].set_yticks([])
        fig.colorbar(im1, ax=axes[1], label="Attribution Score")

        plt.tight_layout()

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            return output_path

        plt.close(fig)
        return None

    @staticmethod
    def export_single(result: ExplanationResult, output_dir: str) -> dict[str, str]:
        """Export JSON and PNG for a single result."""
        import torch

        # clean sample_id
        sid = result.sample_id
        if isinstance(sid, torch.Tensor):
            sid = sid.item()
        clean_id = os.path.basename(str(sid))

        sample_dir = os.path.join(output_dir, clean_id)
        os.makedirs(sample_dir, exist_ok=True)

        json_path = os.path.join(sample_dir, "explanation.json")
        try:
            payload = {
                "sample_id": clean_id,
                "prediction": result.prediction,
                "target": result.target,
                "node_importance": result.node_importance.tolist(),
                "edge_importance": result.edge_importance.tolist(),
                "edge_index": result.edge_index.tolist(),
                "summary_statistics": result.summary_statistics,
                "metadata": result.metadata,
            }
            with open(json_path, "w") as fp:
                json.dump(payload, fp, indent=2)
        except Exception as exc:
            raise ExportError(json_path, str(exc)) from exc

        png_path = os.path.join(sample_dir, "heatmap.png")
        try:
            Exporter.visualize(result, png_path)
        except Exception as exc:
            raise ExportError(png_path, str(exc)) from exc

        return {"json": json_path, "png": png_path}

    @staticmethod
    def export(results: List[ExplanationResult], output_dir: str) -> None:
        """Persist explanation results to disk."""
        import torch

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "summary.csv")
        csv_rows: List[dict] = []

        for res in results:
            try:
                sid = res.sample_id
                if isinstance(sid, torch.Tensor):
                    sid = sid.item()
                clean_id = os.path.basename(str(sid))
                Exporter.export_single(res, output_dir)

                csv_rows.append(
                    {
                        "sample_id": clean_id,
                        "prediction": res.prediction,
                        "target": res.target,
                        "prediction_error": res.summary_statistics.get(
                            "prediction_error", ""
                        ),
                        "top_node_idx": res.summary_statistics.get("top_node_idx", ""),
                        "top_node_score": res.summary_statistics.get(
                            "top_node_score", ""
                        ),
                        "mean_node_importance": res.summary_statistics.get(
                            "mean_node_importance", ""
                        ),
                        "top_edge_idx": res.summary_statistics.get("top_edge_idx", ""),
                        "top_edge_score": res.summary_statistics.get(
                            "top_edge_score", ""
                        ),
                        "mean_edge_importance": res.summary_statistics.get(
                            "mean_edge_importance", ""
                        ),
                        "method": res.metadata.get("method", "unknown"),
                    }
                )
            except Exception as e:
                logger.error(f"Failed to export sample {res.sample_id}: {e}")
                continue

        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            try:
                with open(csv_path, "w", newline="") as fp:
                    writer = csv.DictWriter(fp, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_rows)
                logger.info(f"Exported {len(csv_rows)} explanations to {output_dir}")
            except OSError as exc:
                raise ExportError(csv_path, str(exc)) from exc
