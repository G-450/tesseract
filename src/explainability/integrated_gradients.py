"""Integrated Gradients explainability method via Captum.

Uses Captum's IntegratedGradients to attribute formation energy predictions
back to node embeddings and edge features.

Note:
    Outputs are model-level interpretations, not physical causal evidence
    (PRD §82, §101).
"""

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients  # type: ignore
from torch_geometric.data import Data  # type: ignore

from src.explainability.base import BaseExplainer, ExplanationResult
from src.explainability.exceptions import SampleExplanationError
from src.explainability.exporter import Exporter

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer(BaseExplainer):
    """Explains CGCNN predictions using Captum's Integrated Gradients.

    Computes attribution scores for each atom (node) and each bond (edge)
    by integrating gradients from a baseline (zero embedding) to the
    actual input embedding.

    Note:
        Outputs are model-level interpretations, not physical causal
        evidence (PRD §82, §101).
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the IntegratedGradientsExplainer.

        Args:
            model: A trained CGCNN ``CGCNNLightningModule``. The model is
                immediately frozen (eval mode, no grad) to prevent accidental
                weight modification during attribution.
        """
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Keep a direct reference to the embedding layer so we can hook
        # after it; IG operates on the continuous embedding space rather
        # than on discrete atomic numbers.
        self._embedding_layer = self.model.model.embedding  # type: ignore

    def explain(self, graph: Data, sample_id: Union[int, str]) -> ExplanationResult:
        """Compute Integrated Gradients attributions for a single crystal.

        Args:
            graph: A PyTorch Geometric ``Data`` object. Must have a ``batch``
                attribute; if absent, the graph is treated as a single-graph batch.
            sample_id: Unique identifier used when exporting results.

        Returns:
            ExplanationResult with per-atom and per-bond importance scores.

        Raises:
            SampleExplanationError: If attribution fails for this sample.
        """
        try:
            # Fix issue 1: move graph to model's device and issue 16: use a copy
            graph = graph.clone()
            graph = graph.to(next(self.model.parameters()).device)
            if not hasattr(graph, "batch") or graph.batch is None:
                graph.batch = torch.zeros(
                    graph.x.size(0), dtype=torch.long, device=graph.x.device
                )

            # Define forward function inside so we can capture graph.edge_index
            # without passing it as additional_forward_args (which Captum expands)
            def _forward_from_embedding(
                x_emb: torch.Tensor, e_attr: torch.Tensor
            ) -> torch.Tensor:
                outs = []
                for i in range(x_emb.size(0)):
                    v = x_emb[i]
                    e = e_attr[i]
                    for conv in self.model.model.convs:  # type: ignore
                        v = conv(v, graph.edge_index, e)
                    c = self.model.model.pool(v, graph.batch)  # type: ignore
                    c = self.model.model.fc_pre(c)  # type: ignore
                    c = self.model.model.softplus(c)  # type: ignore
                    outs.append(self.model.model.fc_out(c))  # type: ignore
                return torch.cat(outs, dim=0)

            ig = IntegratedGradients(_forward_from_embedding)

            x_embedded = self._embedding_layer(graph.x.long().view(-1))  # type: ignore
            edge_attr = graph.edge_attr.float()

            node_attr, edge_attr_attr = ig.attribute(
                inputs=(x_embedded.unsqueeze(0), edge_attr.unsqueeze(0)),
                n_steps=50,
            )

            # Reduce over the feature dimension → scalar per atom / bond
            node_importance = node_attr.squeeze(0).sum(dim=-1).detach()
            edge_importance = edge_attr_attr.squeeze(0).sum(dim=-1).detach()

            with torch.no_grad():
                pred = _forward_from_embedding(
                    x_embedded.unsqueeze(0), edge_attr.unsqueeze(0)
                ).squeeze(0)

            return ExplanationResult(
                sample_id=sample_id,
                prediction=float(pred.item()),
                target=float(graph.y.item()),
                node_importance=node_importance,
                edge_importance=edge_importance,
                edge_index=graph.edge_index,
                metadata={"method": "integrated_gradients", "n_steps": 50},
            )

        except Exception as exc:
            raise SampleExplanationError(
                sample_id=sample_id,
                reason=str(exc),
            ) from exc

    def explain_batch(self, graphs: List[Data]) -> List[ExplanationResult]:
        """Compute attributions for multiple crystal graphs.

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
                results.append(self.explain(graph, sample_id))
            except SampleExplanationError as exc:
                logger.warning(str(exc))
        return results

    def visualize(
        self,
        result: ExplanationResult,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a heatmap figure for node and edge importances.

        Args:
            result: The explanation result to visualize.
            output_path: Optional path to save the heatmap.

        Returns:
            The path to the saved heatmap if output_path is provided, else None.
        """
        return Exporter.visualize(result, output_path)

    def export(self, result: ExplanationResult, output_dir: str) -> Dict[str, str]:
        """Export JSON data and PNG heatmap for a single explanation.

        Args:
            result: The explanation result to export.
            output_dir: The directory to save the exported files.

        Returns:
            A dictionary mapping file descriptions to their saved paths.
        """
        return Exporter.export_single(result, output_dir)
