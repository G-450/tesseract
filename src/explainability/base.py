"""Base interfaces for all explainability methods.

Explainability results shall always be interpreted as model explanations
rather than physical proof (PRD §82, §101).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from torch_geometric.data import Data  # type: ignore


@dataclass
class ExplanationResult:
    """Holds the output of one explainability pass on a single crystal graph.

    Note:
        These scores represent model-internal attributions, not physical
        causality (PRD §82). They should be interpreted as explanations
        of what the model learned, not as experimental evidence.

    Attributes:
        sample_id: Identifier for the explained sample (dataset record ID or index).
        prediction: Scalar formation energy predicted by the model (eV/atom).
        target: Ground-truth formation energy value (eV/atom).
        node_importance: Per-atom attribution scores, shape ``[num_nodes]``.
        edge_importance: Per-bond attribution scores, shape ``[num_edges]``.
        edge_index: Edge connectivity tensor, shape ``[2, num_edges]``.
        metadata: Arbitrary key-value pairs (e.g. method name, n_steps).
        summary_statistics: Aggregated statistics derived from importance scores.
    """

    sample_id: Union[int, str]
    prediction: float
    target: float
    node_importance: torch.Tensor  # shape [num_nodes]
    edge_importance: torch.Tensor  # shape [num_edges]
    edge_index: torch.Tensor  # shape [2, num_edges]
    metadata: Dict[str, object] = field(default_factory=dict)
    summary_statistics: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        """Compute summary statistics after field initialization.

        Guards against edge cases such as empty tensors (no atoms/bonds) or
        single-element tensors where ``torch.std`` returns ``nan``.
        """
        node_imp = self.node_importance.float()
        edge_imp = self.edge_importance.float()

        def _safe_abs_argmax(t: torch.Tensor) -> int:
            return int(t.abs().argmax().item()) if t.numel() > 0 else -1

        def _safe_scalar(t: torch.Tensor, fn: str) -> float:
            if t.numel() == 0:
                return 0.0
            val = getattr(t, fn)()
            return float(val.item()) if not (val != val) else 0.0  # NaN → 0

        def _score_at_idx(t: torch.Tensor, idx: int) -> float:
            if idx < 0 or t.numel() == 0:
                return 0.0
            val = t[idx]
            return float(val.item()) if not (val != val) else 0.0

        def _safe_std(t: torch.Tensor) -> float:
            if t.numel() < 2:
                return 0.0
            val = t.std()
            return float(val.item()) if not (val != val) else 0.0

        top_node_idx = _safe_abs_argmax(node_imp)
        top_edge_idx = _safe_abs_argmax(edge_imp)

        self.summary_statistics = {
            "top_node_idx": top_node_idx,
            "top_node_score": _score_at_idx(node_imp, top_node_idx),
            "mean_node_importance": _safe_scalar(node_imp, "mean"),
            "std_node_importance": _safe_std(node_imp),
            "top_edge_idx": top_edge_idx,
            "top_edge_score": _score_at_idx(edge_imp, top_edge_idx),
            "mean_edge_importance": _safe_scalar(edge_imp, "mean"),
            "std_edge_importance": _safe_std(edge_imp),
            "prediction_error": float(abs(self.prediction - self.target)),
        }


class BaseExplainer(ABC):
    """Abstract base class that every explainability method must implement.

    Implementors must override :meth:`explain`, :meth:`explain_batch`,
    :meth:`visualize`, and :meth:`export`.

    The plugin architecture (PRD §83) ensures that training code never
    directly depends on a concrete explainer implementation.

    Note:
        All outputs produced by subclasses are model-level interpretations
        and must not be presented as scientific causal evidence (PRD §82).
    """

    @abstractmethod
    def explain(
        self, graph: Data, sample_id: Union[int, str]
    ) -> Optional[ExplanationResult]:
        """Explain a single crystal graph.

        Args:
            graph: A PyTorch Geometric ``Data`` object representing the crystal.
            sample_id: A unique identifier for the sample (used in exports).

        Returns:
            ExplanationResult containing node/edge attributions and summary stats,
            or ``None`` if the explainer is not supported for this sample.

        Raises:
            SampleExplanationError: If attribution computation fails for this sample.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement explain()."
        )

    @abstractmethod
    def explain_batch(self, graphs: List[Data]) -> List[ExplanationResult]:
        """Explain a collection of crystal graphs.

        Individual sample failures must be logged and skipped rather than
        terminating the entire batch (PRD §88).

        Args:
            graphs: A list of PyTorch Geometric ``Data`` objects.

        Returns:
            A list of :class:`ExplanationResult` objects (may be shorter than
            ``graphs`` if some samples fail).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement explain_batch()."
        )

    @abstractmethod
    def visualize(
        self,
        result: ExplanationResult,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a visualization for an explanation result.

        Produces a figure highlighting node and edge importances on the graph.

        Args:
            result: The :class:`ExplanationResult` to visualize.
            output_path: If provided, save the figure to this path.

        Returns:
            The path where the figure was saved, or ``None`` if no path was given.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement visualize()."
        )

    @abstractmethod
    def export(self, result: ExplanationResult, output_dir: str) -> Dict[str, str]:
        """Export an explanation result to files on disk.

        Args:
            result: The :class:`ExplanationResult` to export.
            output_dir: Directory in which to write artifacts.

        Returns:
            A dict mapping artifact type to absolute file path,
            e.g. ``{"json": "/path/to/explanation.json",
            "png": "/path/to/heatmap.png"}``.

        Raises:
            ExportError: If writing any artifact fails.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement export().")
