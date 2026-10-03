"""Unit and integration tests for Phase 7 model evaluation and reporting framework."""

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

from src.cli.evaluate_cli import main as evaluate_cli_main
from src.config.model_config import CGCNNConfig
from src.evaluation.evaluator import EvaluationResult, Evaluator
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_bias,
    compute_mae,
    compute_r2,
    compute_rmse,
)
from src.evaluation.reporter import PUBLISHED_CGCNN_BASELINE_MAE, Reporter
from src.models.lightning_module import CGCNNLightningModule
from src.visualization.plots import (
    generate_all_plots,
    plot_loss_curves,
    plot_parity,
    plot_residuals,
)

# ---------------------------------------------------------------------------
# Metrics Unit Tests
# ---------------------------------------------------------------------------


def test_compute_mae_known_values() -> None:
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([1.5, 1.5, 3.5, 3.0])
    # |1-1.5| + |2-1.5| + |3-3.5| + |4-3| = 0.5 + 0.5 + 0.5 + 1.0 = 2.5 / 4 = 0.625
    mae = compute_mae(preds, targets)
    assert mae == pytest.approx(0.625, rel=1e-5)


def test_compute_rmse_known_values() -> None:
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([2.0, 2.0, 5.0, 4.0])
    # (1-2)^2 + (2-2)^2 + (3-5)^2 + (4-4)^2 = 1+0+4+0 = 5/4 = 1.25 => sqrt=1.118
    rmse = compute_rmse(preds, targets)
    assert rmse == pytest.approx(np.sqrt(1.25), rel=1e-5)


def test_compute_r2_values() -> None:
    # Perfect predictions
    perfect_preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_r2(perfect_preds, targets) == pytest.approx(1.0, rel=1e-5)

    # Predicting the mean target gives R2 = 0
    mean_val = float(torch.mean(targets))
    mean_preds = torch.full_like(targets, mean_val)
    assert compute_r2(mean_preds, targets) == pytest.approx(0.0, abs=1e-5)

    # Constant target edge case
    const_targets = torch.tensor([3.0, 3.0, 3.0])
    const_preds = torch.tensor([3.0, 3.0, 3.0])
    assert compute_r2(const_preds, const_targets) == pytest.approx(1.0)


def test_compute_bias_values() -> None:
    # Overpredicting by 0.5 on average
    preds = torch.tensor([1.5, 2.5, 3.5])
    targets = torch.tensor([1.0, 2.0, 3.0])
    assert compute_bias(preds, targets) == pytest.approx(0.5, rel=1e-5)

    # Underpredicting by 0.2 on average
    preds_under = torch.tensor([0.8, 1.8, 2.8])
    assert compute_bias(preds_under, targets) == pytest.approx(-0.2, rel=1e-5)


def test_compute_all_metrics() -> None:
    preds = torch.tensor([2.0, 3.0, 4.0])
    targets = torch.tensor([1.0, 3.0, 5.0])
    metrics = compute_all_metrics(preds, targets)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "bias" in metrics
    assert metrics["mae"] == pytest.approx(2.0 / 3.0, rel=1e-5)
    assert metrics["bias"] == pytest.approx(0.0, abs=1e-5)


def test_metrics_invalid_inputs() -> None:
    # Type error
    with pytest.raises(TypeError):
        compute_mae([1.0, 2.0], torch.tensor([1.0, 2.0]))  # type: ignore

    # Empty tensor error
    with pytest.raises(ValueError, match="empty"):
        compute_mae(torch.tensor([]), torch.tensor([]))

    # Shape mismatch error
    with pytest.raises(ValueError, match="does not match"):
        compute_mae(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# EvaluationResult & Reporter Tests
# ---------------------------------------------------------------------------


def test_evaluation_result_attributes() -> None:
    preds = torch.tensor([1.2, 2.2, 3.2])
    targets = torch.tensor([1.0, 2.0, 3.0])
    metrics = {"mae": 0.2, "rmse": 0.2, "r2": 0.99, "bias": 0.2}
    result = EvaluationResult(
        predictions=preds,
        targets=targets,
        metrics=metrics,
        sample_ids=["id_1", "id_2", "id_3"],
    )

    assert len(result.residuals) == 3
    assert torch.allclose(result.residuals, torch.tensor([0.2, 0.2, 0.2]), atol=1e-5)
    assert torch.allclose(
        result.absolute_errors, torch.tensor([0.2, 0.2, 0.2]), atol=1e-5
    )
    assert result.sample_ids == ["id_1", "id_2", "id_3"]


def test_reporter_export(tmp_path: Any) -> None:
    output_dir = tmp_path / "eval_output"
    preds = torch.tensor([0.1, 0.5, 1.2, 2.0, 3.1])
    targets = torch.tensor([0.12, 0.48, 1.15, 2.05, 3.0])
    metrics = compute_all_metrics(preds, targets)
    result = EvaluationResult(
        predictions=preds,
        targets=targets,
        metrics=metrics,
        sample_ids=[101, 102, 103, 104, 105],
    )

    reporter = Reporter()
    paths = reporter.export(result, str(output_dir))

    assert os.path.exists(paths["metrics_json"])
    assert os.path.exists(paths["predictions_csv"])
    assert os.path.exists(paths["summary_md"])

    # Verify metrics.json
    with open(paths["metrics_json"], "r") as f:
        data = json.load(f)
    assert data["num_samples"] == 5
    assert "mae" in data
    assert "benchmark_comparison" in data
    assert (
        data["benchmark_comparison"]["baseline_mae_ev_atom"]
        == PUBLISHED_CGCNN_BASELINE_MAE
    )

    # Verify predictions.csv
    df = pd.read_csv(paths["predictions_csv"])
    assert len(df) == 5
    assert list(df.columns) == [
        "sample_id",
        "true_target",
        "predicted_target",
        "residual",
        "absolute_error",
    ]
    assert df["sample_id"].tolist() == [101, 102, 103, 104, 105]

    # Verify summary.md
    with open(paths["summary_md"], "r") as f:
        summary_text = f.read()
    assert "CGCNN Model Evaluation Summary" in summary_text
    assert "Performance Metrics" in summary_text
    assert "Benchmark Comparison" in summary_text
    assert "Error Distribution" in summary_text


# ---------------------------------------------------------------------------
# Visualization Plots Tests
# ---------------------------------------------------------------------------


def test_plot_parity_and_residuals(tmp_path: Any) -> None:
    preds = torch.randn(50)
    targets = torch.randn(50)
    metrics = compute_all_metrics(preds, targets)

    parity_file = tmp_path / "parity.png"
    residual_file = tmp_path / "residual.png"

    path1 = plot_parity(preds, targets, str(parity_file), metrics=metrics)
    assert os.path.exists(path1)
    assert os.path.getsize(path1) > 0

    path2 = plot_residuals(preds, targets, str(residual_file))
    assert os.path.exists(path2)
    assert os.path.getsize(path2) > 0


def test_plot_loss_curves(tmp_path: Any) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    csv_file = log_dir / "metrics.csv"

    df = pd.DataFrame(
        {
            "epoch": [0, 1, 2, 3],
            "train_loss": [0.5, 0.4, 0.3, 0.2],
            "val_loss": [0.55, 0.42, 0.35, 0.28],
            "val_mae": [0.4, 0.35, 0.3, 0.25],
        }
    )
    df.to_csv(csv_file, index=False)

    loss_png = tmp_path / "loss_curve.png"
    res_path = plot_loss_curves(str(log_dir), str(loss_png))
    assert res_path is not None
    assert os.path.exists(res_path)


def test_generate_all_plots(tmp_path: Any) -> None:
    result = EvaluationResult(
        predictions=torch.randn(20),
        targets=torch.randn(20),
        metrics={"mae": 0.05, "rmse": 0.08, "r2": 0.95, "bias": 0.01},
        sample_ids=list(range(20)),
    )
    plot_dir = tmp_path / "plots"
    saved = generate_all_plots(result, str(plot_dir))
    assert "parity_plot" in saved
    assert "residual_plot" in saved
    assert os.path.exists(saved["parity_plot"])
    assert os.path.exists(saved["residual_plot"])


# ---------------------------------------------------------------------------
# Evaluator Mock Integration Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_test_dataloader() -> DataLoader:
    # 2 batches with 2 graphs each
    # Graph 1
    d1 = Data(
        x=torch.tensor([8, 14, 26]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.randn(4, 41),
        y=torch.tensor([1.2]),
    )
    d1.dataset_record_id = 1001

    # Graph 2
    d2 = Data(
        x=torch.tensor([1, 92]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.randn(2, 41),
        y=torch.tensor([0.8]),
    )
    d2.dataset_record_id = 1002

    # Graph 3
    d3 = Data(
        x=torch.tensor([6, 8]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.randn(2, 41),
        y=torch.tensor([2.5]),
    )
    d3.dataset_record_id = 1003

    return DataLoader([d1, d2, d3], batch_size=2, shuffle=False)


def test_evaluator_with_mock_model_and_dataloader(
    mock_test_dataloader: DataLoader, tmp_path: Any
) -> None:
    config = CGCNNConfig()
    config.model.hidden_dim = 32
    config.model.num_conv_layers = 2
    model = CGCNNLightningModule(config)
    model.eval()

    evaluator = Evaluator(
        checkpoint_path=str(tmp_path / "mock.ckpt"),
        device="cpu",
    )

    result = evaluator.evaluate_dataloader(mock_test_dataloader, model=model)

    assert isinstance(result, EvaluationResult)
    assert len(result.predictions) == 3
    assert len(result.targets) == 3
    assert len(result.sample_ids) == 3
    assert result.sample_ids == [1001, 1002, 1003]
    assert "mae" in result.metrics
    assert "rmse" in result.metrics
    assert "r2" in result.metrics
    assert "bias" in result.metrics
    assert not torch.isnan(result.residuals).any()


def test_evaluate_cli_flow(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create mock checkpoint
    ckpt_path = tmp_path / "mock.ckpt"
    config = CGCNNConfig()
    config.model.hidden_dim = 16
    config.model.num_conv_layers = 2
    module = CGCNNLightningModule(config)
    torch.save(
        {"state_dict": module.state_dict(), "hyper_parameters": {"config": config}},
        ckpt_path,
    )

    # Mock Evaluator.run
    mock_result = EvaluationResult(
        predictions=torch.tensor([1.0, 2.0, 3.0]),
        targets=torch.tensor([1.1, 1.9, 3.2]),
        metrics={"mae": 0.133, "rmse": 0.152, "r2": 0.98, "bias": -0.067},
        sample_ids=[1, 2, 3],
    )

    monkeypatch.setattr(Evaluator, "run", lambda self: mock_result)

    output_dir = tmp_path / "eval_out"
    test_args = [
        "evaluate_cli.py",
        "--checkpoint",
        str(ckpt_path),
        "--output-dir",
        str(output_dir),
    ]

    monkeypatch.setattr("sys.argv", test_args)
    evaluate_cli_main()

    assert os.path.exists(output_dir / "metrics.json")
    assert os.path.exists(output_dir / "predictions.csv")
    assert os.path.exists(output_dir / "summary.md")
    assert os.path.exists(output_dir / "parity_plot.png")
    assert os.path.exists(output_dir / "residual_plot.png")
