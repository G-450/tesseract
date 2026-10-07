"""Unit and integration tests for Phase 8 explainability subsystem.

Tests cover:
  - BaseExplainer ABC enforcement
  - ExplanationResult dataclass and summary statistics
  - IntegratedGradientsExplainer forward pass
  - GNNExplainerWrapper graceful fallback
  - Exporter artifact generation
  - Exception hierarchy
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import pytest
import torch
from torch_geometric.data import Data  # type: ignore

from src.config.model_config import CGCNNConfig
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
from src.models.lightning_module import CGCNNLightningModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph() -> Data:
    """Return a minimal valid crystal graph with 3 atoms and 4 bonds."""
    return Data(
        x=torch.tensor([1, 2, 3], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.randn(4, 41),
        y=torch.tensor([[1.0]]),
        batch=torch.zeros(3, dtype=torch.long),
    )


def _make_model(hidden_dim: int = 16, num_conv_layers: int = 2) -> CGCNNLightningModule:
    """Return a small CGCNN model for testing."""
    config = CGCNNConfig()
    config.model.hidden_dim = hidden_dim
    config.model.num_conv_layers = num_conv_layers
    return CGCNNLightningModule(config)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_exception_hierarchy() -> None:
    """All explainability exceptions must inherit from ExplainabilityError."""
    assert issubclass(ExplainerInitializationError, ExplainabilityError)
    assert issubclass(SampleExplanationError, ExplainabilityError)
    assert issubclass(ExportError, ExplainabilityError)


def test_sample_explanation_error_attrs() -> None:
    exc = SampleExplanationError(sample_id="mp-001", reason="NaN in gradient")
    assert exc.sample_id == "mp-001"
    assert exc.reason == "NaN in gradient"
    assert "mp-001" in str(exc)


def test_export_error_attrs() -> None:
    exc = ExportError(path="/tmp/foo.json", reason="disk full")
    assert exc.path == "/tmp/foo.json"
    assert "foo.json" in str(exc)


# ---------------------------------------------------------------------------
# BaseExplainer ABC enforcement
# ---------------------------------------------------------------------------


def test_base_explainer_cannot_be_instantiated_directly() -> None:
    """BaseExplainer is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseExplainer()  # type: ignore


def test_base_explainer_partial_implementation_raises() -> None:
    """Partial implementations are rejected by ABC."""

    class PartialExplainer(BaseExplainer):
        def explain(self, graph: Data, sample_id: Union[int, str]) -> ExplanationResult:
            raise NotImplementedError

        # Missing: explain_batch, visualize, export

    with pytest.raises(TypeError):
        PartialExplainer()  # type: ignore


def test_base_explainer_full_implementation_accepted() -> None:
    """A class implementing all abstract methods can be instantiated."""

    class ConcreteExplainer(BaseExplainer):
        def explain(
            self, graph: Data, sample_id: Union[int, str]
        ) -> ExplanationResult:  # type: ignore[override]
            raise NotImplementedError

        def explain_batch(self, graphs: List[Data]) -> List[ExplanationResult]:
            return []

        def visualize(
            self, result: ExplanationResult, output_path: Optional[str] = None
        ) -> Optional[str]:
            return None

        def export(self, result: ExplanationResult, output_dir: str) -> Dict[str, str]:
            return {}

    explainer = ConcreteExplainer()
    assert explainer is not None


# ---------------------------------------------------------------------------
# ExplanationResult dataclass
# ---------------------------------------------------------------------------


def test_explanation_result_summary_statistics() -> None:
    """ExplanationResult.__post_init__ must populate summary_statistics."""
    node_imp = torch.tensor([0.1, -0.9, 0.3])
    edge_imp = torch.tensor([0.5, 0.2, -0.8, 0.1])

    result = ExplanationResult(
        sample_id="test-01",
        prediction=1.5,
        target=1.0,
        node_importance=node_imp,
        edge_importance=edge_imp,
        edge_index=torch.zeros((2, 4), dtype=torch.long),
        metadata={"method": "integrated_gradients"},
    )

    stats = result.summary_statistics
    assert "top_node_idx" in stats
    assert stats["top_node_idx"] == 1  # index of max abs (-0.9)
    assert stats["top_node_score"] == pytest.approx(-0.9, rel=1e-4)
    assert "top_edge_idx" in stats
    assert stats["top_edge_idx"] == 2  # index of max abs (-0.8)
    assert stats["top_edge_score"] == pytest.approx(-0.8, rel=1e-4)
    assert stats["prediction_error"] == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# IntegratedGradientsExplainer
# ---------------------------------------------------------------------------


def test_integrated_gradients_explain_output_shapes() -> None:
    """IG explain() must return correct tensor shapes."""
    model = _make_model()
    explainer = IntegratedGradientsExplainer(model)
    graph = _make_graph()

    result = explainer.explain(graph, sample_id="ig-01")

    assert isinstance(result, ExplanationResult)
    assert result.sample_id == "ig-01"
    assert result.node_importance.shape == (graph.num_nodes,)
    assert result.edge_importance.shape == (graph.num_edges,)
    assert result.edge_index.shape == (2, graph.num_edges)
    assert result.metadata["method"] == "integrated_gradients"
    assert "top_node_idx" in result.summary_statistics


def test_integrated_gradients_explain_batch_skips_bad_samples() -> None:
    """explain_batch must skip failing graphs and not raise."""
    model = _make_model()
    explainer = IntegratedGradientsExplainer(model)

    good_graph = _make_graph()
    # Corrupt a graph: out-of-bounds node in edge_index -> will fail in conv
    bad_graph = Data(
        x=torch.tensor([1, 2], dtype=torch.long),
        edge_index=torch.tensor([[0, 5], [5, 0]], dtype=torch.long),
        edge_attr=torch.randn((2, 41)),
        y=torch.tensor([[0.0]]),
        batch=torch.zeros(2, dtype=torch.long),
    )

    results = explainer.explain_batch([good_graph, bad_graph])
    assert len(results) == 1
    assert results[0].node_importance.shape[0] == good_graph.num_nodes


def test_integrated_gradients_explain_batch_returns_list() -> None:
    """explain_batch returns a list even with a single valid graph."""
    model = _make_model()
    explainer = IntegratedGradientsExplainer(model)
    graph = _make_graph()

    results = explainer.explain_batch([graph])
    assert isinstance(results, list)
    assert len(results) == 1


def test_integrated_gradients_visualize_returns_path(tmp_path: Any) -> None:
    """visualize() must save a file and return its path."""
    model = _make_model()
    explainer = IntegratedGradientsExplainer(model)
    graph = _make_graph()
    result = explainer.explain(graph, sample_id="vis-01")

    out = str(tmp_path / "heatmap.png")
    returned = explainer.visualize(result, output_path=out)

    assert returned == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_integrated_gradients_export_creates_files(tmp_path: Any) -> None:
    """export() must produce both JSON and PNG artifacts."""
    model = _make_model()
    explainer = IntegratedGradientsExplainer(model)
    graph = _make_graph()
    result = explainer.explain(graph, sample_id="exp-01")

    paths = explainer.export(result, str(tmp_path))

    assert "json" in paths
    assert "png" in paths
    assert os.path.exists(paths["json"])
    assert os.path.exists(paths["png"])

    with open(paths["json"]) as fp:
        data = json.load(fp)
    assert data["sample_id"] == "exp-01"
    assert "summary_statistics" in data
    assert "node_importance" in data
    assert "edge_importance" in data


# ---------------------------------------------------------------------------
# GNNExplainerWrapper
# ---------------------------------------------------------------------------


def test_gnn_explainer_wrapper_graceful_fallback(monkeypatch: Any) -> None:
    """GNNExplainerWrapper must not crash even when unsupported."""
    import src.explainability.gnn_explainer as ge

    monkeypatch.setattr(ge, "_HAS_GNN_EXPLAINER", False)

    model = _make_model()
    wrapper = GNNExplainerWrapper(model)

    assert wrapper.is_supported is False
    graph = _make_graph()

    result = wrapper.explain(graph, sample_id=1)
    assert result is None

    results = wrapper.explain_batch([graph])
    assert results == []


def test_gnn_explainer_batch_returns_list() -> None:
    """explain_batch must return a list regardless of support status."""
    model = _make_model()
    wrapper = GNNExplainerWrapper(model)
    graph = _make_graph()

    results = wrapper.explain_batch([graph])
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


def test_exporter_creates_directory_structure(tmp_path: Any) -> None:
    """Exporter must create per-sample directories and a summary CSV."""
    results = [
        ExplanationResult(
            sample_id="mat-001",
            prediction=1.5,
            target=1.0,
            node_importance=torch.randn(5),
            edge_importance=torch.randn(8),
            edge_index=torch.zeros((2, 8), dtype=torch.long),
            metadata={"method": "integrated_gradients"},
        ),
        ExplanationResult(
            sample_id="mat-002",
            prediction=-0.3,
            target=-0.5,
            node_importance=torch.randn(3),
            edge_importance=torch.randn(4),
            edge_index=torch.zeros((2, 4), dtype=torch.long),
            metadata={"method": "integrated_gradients"},
        ),
    ]

    Exporter.export(results, str(tmp_path))

    # Per-material artifacts
    for sid in ["mat-001", "mat-002"]:
        assert os.path.exists(tmp_path / sid / "explanation.json")
        assert os.path.exists(tmp_path / sid / "heatmap.png")

    # Summary CSV
    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists()

    import csv

    with open(csv_path, newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "mat-001"
    assert "top_node_idx" in rows[0]
    assert "top_edge_score" in rows[0]


def test_exporter_json_contains_summary_statistics(tmp_path: Any) -> None:
    """Per-material JSON must include summary_statistics field."""
    result = ExplanationResult(
        sample_id=42,
        prediction=2.1,
        target=2.0,
        node_importance=torch.tensor([0.3, 0.7, 0.1]),
        edge_importance=torch.tensor([0.5, 0.2]),
        edge_index=torch.zeros((2, 2), dtype=torch.long),
        metadata={"method": "integrated_gradients"},
    )

    Exporter.export([result], str(tmp_path))

    json_path = tmp_path / "42" / "explanation.json"
    assert json_path.exists()

    with open(json_path) as fp:
        data = json.load(fp)

    assert "summary_statistics" in data
    assert data["summary_statistics"]["top_node_idx"] == 1
    assert data["summary_statistics"]["prediction_error"] == pytest.approx(
        0.1, rel=1e-4
    )


# ---------------------------------------------------------------------------
# CLI Integration
# ---------------------------------------------------------------------------


def test_explain_cli_integration(tmp_path: Any, monkeypatch: Any) -> None:
    """Integration test for explain_cli.py with a mocked Evaluator."""
    import csv
    import sys

    from src.cli.explain_cli import main

    # Mock dataloader
    graph1 = _make_graph()
    graph1.dataset_record_id = torch.tensor([12345])
    graph2 = _make_graph()
    graph2.dataset_record_id = torch.tensor([67890])

    class MockEvaluator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def load_model(self) -> Any:
            return _make_model()

        def get_dataloader(self) -> Any:
            # simulate a batch that has to_data_list
            class MockBatch:
                def to_data_list(self) -> Any:
                    return [graph1, graph2]

            return [MockBatch()]

    monkeypatch.setattr("src.cli.explain_cli.Evaluator", MockEvaluator)

    dummy_ckpt = str(tmp_path / "dummy.ckpt")
    with open(dummy_ckpt, "w") as f:
        f.write("dummy")

    out_dir = str(tmp_path / "explanations")
    test_args = [
        "explain_cli.py",
        "--checkpoint",
        dummy_ckpt,
        "--output-dir",
        out_dir,
        "--num-samples",
        "2",
        "--method",
        "integrated_gradients",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    main()

    # Assert outputs
    assert os.path.exists(os.path.join(out_dir, "12345"))
    assert os.path.exists(os.path.join(out_dir, "12345", "explanation.json"))
    assert os.path.exists(os.path.join(out_dir, "12345", "heatmap.png"))

    assert os.path.exists(os.path.join(out_dir, "67890"))
    assert os.path.exists(os.path.join(out_dir, "67890", "explanation.json"))
    assert os.path.exists(os.path.join(out_dir, "67890", "heatmap.png"))

    csv_path = os.path.join(out_dir, "summary.csv")
    assert os.path.exists(csv_path)

    with open(csv_path, newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "12345"
    assert rows[1]["sample_id"] == "67890"
