"""GNNExplainer wrapper for CGCNN explainability.

Wraps PyTorch Geometric's GNNExplainer behind the BaseExplainer interface.
If the current PyG version does not support ``torch_geometric.explain``, the
explainer gracefully disables itself without crashing the evaluation pipeline
(PRD §85, §88).

Note:
    Outputs are model-level interpretations, not physical causal evidence
    (PRD §82, §101).
"""

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data  # type: ignore

from src.explainability.base import BaseExplainer, ExplanationResult
from src.explainability.exceptions import (
    ExplainerInitializationError,
    SampleExplanationError,
)
from src.explainability.exporter import Exporter

logger = logging.getLogger(__name__)

try:
    from torch_geometric.explain import Explainer, GNNExplainer  # type: ignore

    _HAS_GNN_EXPLAINER = True
except ImportError:
    _HAS_GNN_EXPLAINER = False


class GNNExplainerWrapper(BaseExplainer):
    """Explains CGCNN predictions using PyG's GNNExplainer.

    If GNNExplainer is not available in the installed PyG version, the
    ``is_supported`` attribute is set to ``False`` and calls to :meth:`explain`
    return ``None`` with a warning, allowing the rest of the pipeline to continue
    (PRD §85, §88).

    Note:
        Outputs are model-level interpretations, not physical causal
        evidence (PRD §82, §101).
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the GNNExplainerWrapper.

        Args:
            model: A trained CGCNN ``CGCNNLightningModule``. The model is
                immediately frozen to prevent accidental weight modification.

        Raises:
            ExplainerInitializationError: Logged as a warning only; the
                ``is_supported`` flag is set to ``False`` instead of raising,
                so the pipeline continues (PRD §88).
        """
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.is_supported: bool = False

        if not _HAS_GNN_EXPLAINER:
            logger.warning(
                "torch_geometric.explain is not available in this PyG version. "
                "GNNExplainer will be disabled."
            )
            return

        try:
            self._explainer = Explainer(
                model=self.model,
                algorithm=GNNExplainer(epochs=50),
                explanation_type="model",
                node_mask_type=None,
                edge_mask_type="object",
                model_config=dict(
                    mode="regression",
                    task_level="graph",
                    return_type="raw",
                ),
            )
            self.is_supported = True
            logger.info(
                "GNNExplainer initialized with node_mask_type=None. "
                "Node importance scores will always be zero."
            )
        except Exception as exc:
            logger.warning(
                ExplainerInitializationError(method="GNNExplainer", reason=str(exc))
            )

    def explain(
        self, graph: Data, sample_id: Union[int, str]
    ) -> Optional[ExplanationResult]:
        """Compute GNNExplainer masks for a single crystal.

        Args:
            graph: A PyTorch Geometric ``Data`` object. A ``batch`` tensor is
                added automatically for single-graph inputs.
            sample_id: Unique identifier used when exporting results.

        Returns:
            ExplanationResult with per-atom and per-bond importance masks,
            or ``None`` if GNNExplainer is not supported.

        Raises:
            SampleExplanationError: If mask computation fails for this sample.
        """
        if not self.is_supported:
            logger.warning(f"GNNExplainer is disabled. Skipping sample '{sample_id}'.")
            return None

        try:
            # Fix issue 1: move graph to model's device and issue 16: use a copy
            graph = graph.clone()
            graph = graph.to(next(self.model.parameters()).device)
            if not hasattr(graph, "batch") or graph.batch is None:
                graph.batch = torch.zeros(
                    graph.x.size(0), dtype=torch.long, device=graph.x.device
                )

            with torch.no_grad():
                pred = self.model(
                    graph.x, graph.edge_index, graph.edge_attr, graph.batch
                )

            explanation = self._explainer(
                graph.x,
                graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=graph.batch,
            )

            if hasattr(explanation, "node_mask") and explanation.node_mask is not None:
                node_importance = explanation.node_mask.sum(dim=-1).detach()
            else:
                node_importance = torch.zeros(graph.num_nodes, device=graph.x.device)

            if hasattr(explanation, "edge_mask") and explanation.edge_mask is not None:
                edge_importance = explanation.edge_mask.detach()
                if edge_importance.dim() > 1:
                    edge_importance = edge_importance.sum(dim=-1)
            else:
                edge_importance = torch.zeros(graph.num_edges, device=graph.x.device)

            return ExplanationResult(
                sample_id=sample_id,
                prediction=float(pred.item()),
                target=float(graph.y.item()),
                node_importance=node_importance,
                edge_importance=edge_importance,
                edge_index=graph.edge_index,
                metadata={
                    "method": "gnn_explainer",
                    "epochs": 50,
                    "note": (
                        "node_mask_type=None, node importance is intentionally zero."
                    ),
                },
            )

        except Exception as exc:
            raise SampleExplanationError(
                sample_id=sample_id,
                reason=str(exc),
            ) from exc

    def explain_batch(self, graphs: List[Data]) -> List[ExplanationResult]:
        """Compute GNNExplainer masks for multiple crystal graphs.

        Individual failures are logged and skipped so that the rest of the
        batch continues (PRD §88).

        Args:
            graphs: List of PyTorch Geometric ``Data`` objects.

        Returns:
            List of successful :class:`ExplanationResult` objects.
        """
        results: List[ExplanationResult] = []
        for i, graph in enumerate(graphs):
            sample_id = getattr(graph, "sample_id", i)
            try:
                result = self.explain(graph, sample_id)
                if result is not None:
                    results.append(result)
            except SampleExplanationError as exc:
                logger.warning(str(exc))
        return results

    def visualize(
        self,
        result: ExplanationResult,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a heatmap figure for node and edge importances."""
        return Exporter.visualize(result, output_path)

    def export(self, result: ExplanationResult, output_dir: str) -> Dict[str, str]:
        """Export JSON data and PNG heatmap for a single explanation."""
        return Exporter.export_single(result, output_dir)
