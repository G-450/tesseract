"""Unit and integration tests for the Phase 10 reporting subsystem.

Tests cover:
  - Experiment configuration loading, validation and run-ID resolution
  - The canonical run directory layout and config snapshotting
  - Provenance collection and round-tripping
  - The publication figures added for PRD §95
  - Training / explainability / prediction artifact summaries
  - Report content completeness and byte-level reproducibility
  - report_cli.py integration, including --report-only
"""

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest
import torch

from src.cli.report_cli import main as report_cli_main
from src.config.exceptions import (
    ConfigFileError,
    ConfigurationError,
    UnknownConfigKeyError,
)
from src.config.experiment_config import (
    ExperimentConfig,
    generate_run_id,
    load_experiment_config,
    sanitize_run_id,
)
from src.evaluation.evaluator import EvaluationResult
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
from src.evaluation.reporter import Reporter
from src.utils.provenance import (
    collect_dataset_provenance,
    collect_git_provenance,
    collect_hardware_provenance,
    collect_library_versions,
    collect_provenance,
    file_sha256,
    load_provenance,
)
from src.utils.run_layout import (
    SUBDIRECTORIES,
    RunDirectoryExistsError,
    RunLayout,
)
from src.visualization.plots import (
    PUBLICATION_RC_PARAMS,
    generate_all_plots,
    plot_learning_rate,
    plot_prediction_histogram,
)

FIXED_TIME = datetime(2026, 10, 10, 15, 30, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(num_samples: int = 50) -> EvaluationResult:
    """Returns a deterministic EvaluationResult for reporting tests."""
    generator = torch.Generator().manual_seed(7)
    targets = torch.randn(num_samples, generator=generator) - 1.4
    predictions = targets + 0.05 * torch.randn(num_samples, generator=generator)
    residuals = predictions - targets
    return EvaluationResult(
        predictions=predictions,
        targets=targets,
        metrics={
            "mae": float(residuals.abs().mean()),
            "rmse": float(residuals.pow(2).mean().sqrt()),
            "r2": 0.997,
            "bias": float(residuals.mean()),
        },
        sample_ids=[f"rec_{index}" for index in range(num_samples)],
    )


def _write_training_log(directory: str, epochs: int = 6, with_lr: bool = True) -> str:
    """Writes a Lightning-style CSV metrics log and returns its path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "metrics.csv")
    header = ["epoch", "step", "train_loss", "val_loss", "val_MAE"]
    if with_lr:
        header.append("lr-Adam")

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for epoch in range(epochs):
            row: List[Any] = [
                epoch,
                epoch * 10,
                round(1.0 / (epoch + 1), 6),
                round(1.1 / (epoch + 1), 6),
                round(0.5 / (epoch + 1), 6),
            ]
            if with_lr:
                row.append(0.001 * (0.5 ** (epoch // 3)))
            writer.writerow(row)
    return path


def _write_explanations(directory: str, sample_ids: List[str]) -> None:
    """Writes explainability artifacts in the exporter's layout."""
    os.makedirs(directory, exist_ok=True)
    rows = []
    for sample_id in sample_ids:
        sample_dir = os.path.join(directory, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        with open(
            os.path.join(sample_dir, "explanation.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump({"sample_id": sample_id}, handle)
        rows.append({"sample_id": sample_id, "method": "integrated_gradients"})

    with open(
        os.path.join(directory, "summary.csv"), "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "method"])
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(directory: str, count: int = 3) -> None:
    """Writes inference artifacts in the prediction exporter's layout."""
    os.makedirs(directory, exist_ok=True)
    with open(
        os.path.join(directory, "predictions.csv"), "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "formation_energy_ev_per_atom"]
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {"sample_id": f"cif_{index}", "formation_energy_ev_per_atom": -1.0}
            )

    with open(
        os.path.join(directory, "predictions.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "predictions": [],
                "summary": {"count": count, "mean": -1.0},
                "failures": [["bad.cif", "unparseable"]],
            },
            handle,
        )


def _write_metrics(layout: RunLayout, result: EvaluationResult) -> Dict[str, Any]:
    """Exports metrics.json for a run and returns the parsed document."""
    Reporter(baseline_mae=0.039).export(result, layout.root)
    with open(layout.metrics_json, "r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)
    return payload


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


def test_experiment_config_defaults_when_file_missing(tmp_path: Any) -> None:
    config = load_experiment_config(str(tmp_path / "absent.yaml"))
    assert config == ExperimentConfig()
    assert config.runs_root == os.path.join("experiments", "runs")


def test_experiment_config_loads_shipped_file() -> None:
    config = load_experiment_config("configs/experiment.yaml")
    assert config.name == "cgcnn_mp_e_form"
    assert config.seed == 42
    assert config.run_id is None
    assert config.baseline_mae_ev_atom == pytest.approx(0.039)
    assert config.logging.level == "INFO"


def test_experiment_config_rejects_unknown_keys(tmp_path: Any) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("experiment:\n  nmae: oops\n", encoding="utf-8")
    with pytest.raises(UnknownConfigKeyError) as excinfo:
        load_experiment_config(str(path))
    assert "nmae" in excinfo.value.unknown_keys
    assert excinfo.value.section == "experiment"


def test_experiment_config_rejects_unknown_nested_keys(tmp_path: Any) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("experiment:\n  logging:\n    verbosity: 2\n", encoding="utf-8")
    with pytest.raises(UnknownConfigKeyError) as excinfo:
        load_experiment_config(str(path))
    assert excinfo.value.section == "experiment.logging"


def test_experiment_config_rejects_malformed_yaml(tmp_path: Any) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("experiment: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigFileError):
        load_experiment_config(str(path))


def test_generate_run_id_is_timestamped_and_safe() -> None:
    run_id = generate_run_id("cgcnn mp/e_form", now=FIXED_TIME)
    assert run_id == "cgcnn_mp_e_form_20261010T153000Z"
    assert os.sep not in run_id


def test_sanitize_run_id_strips_traversal() -> None:
    assert sanitize_run_id("../../etc/passwd") == "etc_passwd"
    with pytest.raises(ConfigurationError):
        sanitize_run_id("../..")


def test_resolve_run_id_precedence() -> None:
    config = ExperimentConfig(run_id="configured")
    assert config.resolve_run_id() == "configured"
    assert config.resolve_run_id("from_cli") == "from_cli"
    assert ExperimentConfig().resolve_run_id().startswith("cgcnn_mp_e_form_")


# ---------------------------------------------------------------------------
# Run layout
# ---------------------------------------------------------------------------


def test_run_layout_creates_every_subdirectory(tmp_path: Any) -> None:
    layout = RunLayout.for_run(str(tmp_path / "runs"), "run_a").create()
    for name in SUBDIRECTORIES:
        assert os.path.isdir(os.path.join(layout.root, name))
    assert layout.root.endswith(os.path.join("runs", "run_a"))


def test_run_layout_refuses_to_reuse_existing_run(tmp_path: Any) -> None:
    layout = RunLayout.for_run(str(tmp_path / "runs"), "run_a").create()
    with pytest.raises(RunDirectoryExistsError):
        layout.create(exist_ok=False)


def test_run_layout_overwrite_clears_previous_artifacts(tmp_path: Any) -> None:
    layout = RunLayout.for_run(str(tmp_path / "runs"), "run_a").create()
    stale = os.path.join(layout.plots_dir, "stale.png")
    with open(stale, "w", encoding="utf-8") as handle:
        handle.write("x")

    layout.create(overwrite=True)
    assert not os.path.exists(stale)
    assert os.path.isdir(layout.plots_dir)


def test_run_layout_snapshots_configs(tmp_path: Any) -> None:
    layout = RunLayout.for_run(str(tmp_path / "runs"), "run_a").create()
    written = layout.snapshot_configs(
        {
            "dataset": "configs/dataset.yaml",
            "model": "configs/model.yaml",
            "experiment": "configs/experiment.yaml",
            "absent": str(tmp_path / "nope.yaml"),
        }
    )
    assert set(written) == {"dataset", "model", "experiment"}
    for path in written.values():
        assert os.path.isfile(path)


def test_run_layout_relative_paths_are_posix(tmp_path: Any) -> None:
    layout = RunLayout.for_run(str(tmp_path / "runs"), "run_a").create()
    assert layout.relative(layout.plots_dir) == "plots"
    assert (
        layout.relative(os.path.join(layout.plots_dir, "parity_plot.png"))
        == "plots/parity_plot.png"
    )
    assert layout.relative(None) is None
    outside = str(tmp_path / "elsewhere.txt")
    assert layout.relative(outside) == outside


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_collect_library_versions_reports_installed_and_missing() -> None:
    versions = collect_library_versions(("torch", "definitely-not-installed"))
    assert versions["torch"] is not None
    assert versions["definitely-not-installed"] is None


def test_collect_hardware_provenance_populates_platform() -> None:
    hardware = collect_hardware_provenance()
    assert hardware.platform
    assert hardware.python_version
    assert isinstance(hardware.cuda_available, bool)


def test_collect_git_provenance_outside_repository(tmp_path: Any) -> None:
    provenance = collect_git_provenance(str(tmp_path))
    # Outside a repository git fails, which must degrade rather than raise.
    assert provenance.commit is None or isinstance(provenance.commit, str)


def test_collect_dataset_provenance_reads_cached_artifacts(tmp_path: Any) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "dataset_fingerprint": "abc123",
                "stats": {"total_records": 10, "valid_records": 9},
            }
        ),
        encoding="utf-8",
    )
    graphs_dir = tmp_path / "processed" / "graphs"
    graphs_dir.mkdir(parents=True)
    (graphs_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": "abc123",
                "successful_graphs": 9,
                "failed_graphs": 1,
                "shards_created": 2,
                "graph_config": {"cutoff_radius": 8.0},
            }
        ),
        encoding="utf-8",
    )

    provenance = collect_dataset_provenance(
        raw_data_path="datasets/raw/mp_e_form.json",
        metadata_dir=str(metadata_dir),
        processed_data_dir=str(tmp_path / "processed"),
        target_column="e_form",
    )
    assert provenance.fingerprint == "abc123"
    assert provenance.total_records == 10
    assert provenance.successful_graphs == 9
    assert provenance.failed_graphs == 1
    assert provenance.graph_config == {"cutoff_radius": 8.0}


def test_collect_dataset_provenance_tolerates_missing_sources(tmp_path: Any) -> None:
    provenance = collect_dataset_provenance(
        metadata_dir=str(tmp_path / "absent"),
        processed_data_dir=str(tmp_path / "absent"),
    )
    assert provenance.fingerprint is None
    assert provenance.total_records is None


def test_file_sha256_round_trip(tmp_path: Any) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"cgcnn")
    digest = file_sha256(str(path))
    assert digest is not None and len(digest) == 64
    assert file_sha256(str(tmp_path / "absent.bin")) is None


def test_provenance_write_and_load_round_trip(tmp_path: Any) -> None:
    provenance = collect_provenance(
        run_id="run_a",
        experiment_name="cgcnn",
        seed=42,
        configs={"experiment": {"seed": 42}},
        now=FIXED_TIME,
    )
    path = provenance.write(str(tmp_path / "metadata.json"))
    reloaded = load_provenance(path)

    assert reloaded.run_id == "run_a"
    assert reloaded.seed == 42
    assert reloaded.generated_at == provenance.generated_at
    assert reloaded.hardware.platform == provenance.hardware.platform
    assert reloaded.to_dict() == provenance.to_dict()


def test_provenance_records_every_prd_required_field(tmp_path: Any) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")

    provenance = collect_provenance(
        run_id="run_a",
        experiment_name="cgcnn",
        seed=42,
        checkpoint_path=str(checkpoint),
        configs={"model": {"hidden_dim": 64}},
        config_snapshots={"model": "config/model.yaml"},
        now=FIXED_TIME,
    )
    # PRD §17: commit, timestamp, seed, config snapshot, library versions,
    # hardware, CUDA, dataset version, model version.
    assert provenance.generated_at == "2026-10-10T15:30:00+00:00"
    assert provenance.seed == 42
    assert provenance.configs["model"]["hidden_dim"] == 64
    assert provenance.config_snapshots["model"] == "config/model.yaml"
    assert provenance.library_versions["torch"] is not None
    assert provenance.hardware.python_version
    assert provenance.model_version == "model.ckpt"
    assert provenance.checkpoint_sha256 is not None
    assert hasattr(provenance.git, "commit")
    assert hasattr(provenance.dataset, "fingerprint")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_publication_style_overrides_matplotlib_defaults() -> None:
    # PRD §97 forbids relying on default styling.
    assert PUBLICATION_RC_PARAMS["savefig.dpi"] == 300
    assert PUBLICATION_RC_PARAMS["axes.labelweight"] == "bold"
    assert PUBLICATION_RC_PARAMS["font.sans-serif"] == ["DejaVu Sans"]


def test_plot_prediction_histogram_writes_png(tmp_path: Any) -> None:
    result = _make_result()
    output = tmp_path / "prediction_histogram.png"
    path = plot_prediction_histogram(
        result.predictions, result.targets, str(output), dpi=72
    )
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


def test_plot_learning_rate_from_log(tmp_path: Any) -> None:
    log_dir = tmp_path / "logs"
    _write_training_log(str(log_dir))
    output = tmp_path / "learning_rate.png"
    path = plot_learning_rate(str(log_dir), str(output), dpi=72)
    assert path is not None and os.path.isfile(path)


def test_plot_learning_rate_returns_none_without_lr_column(tmp_path: Any) -> None:
    log_dir = tmp_path / "logs"
    _write_training_log(str(log_dir), with_lr=False)
    result = plot_learning_rate(str(log_dir), str(tmp_path / "lr.png"), dpi=72)
    assert result is None


def test_plot_learning_rate_returns_none_without_log(tmp_path: Any) -> None:
    assert (
        plot_learning_rate(str(tmp_path / "absent"), str(tmp_path / "lr.png")) is None
    )


def test_generate_all_plots_covers_required_visualizations(tmp_path: Any) -> None:
    result = _make_result()
    log_dir = tmp_path / "logs"
    _write_training_log(str(log_dir))

    saved = generate_all_plots(result, str(tmp_path / "plots"), log_dir=str(log_dir))

    # PRD §95 required figures, minus the evaluation summary which is Markdown.
    assert set(saved) == {
        "parity_plot",
        "residual_plot",
        "prediction_histogram",
        "loss_curve",
        "learning_rate",
    }
    for path in saved.values():
        assert os.path.isfile(path)


def test_generate_all_plots_without_logs_skips_log_figures(tmp_path: Any) -> None:
    saved = generate_all_plots(_make_result(), str(tmp_path / "plots"))
    assert set(saved) == {"parity_plot", "residual_plot", "prediction_histogram"}


# ---------------------------------------------------------------------------
# Artifact summaries
# ---------------------------------------------------------------------------


def test_summarize_training_log_recovers_history(tmp_path: Any) -> None:
    log_dir = tmp_path / "logs" / "csv" / "version_0"
    _write_training_log(str(log_dir))

    summary = summarize_training_log(str(tmp_path / "logs"))
    assert summary.available
    assert summary.epochs_completed == 6
    assert summary.best_val_loss == pytest.approx(1.1 / 6, rel=1e-3)
    assert summary.best_val_loss_epoch == 5
    assert summary.best_val_mae == pytest.approx(0.5 / 6, rel=1e-3)
    assert summary.initial_learning_rate == pytest.approx(0.001)
    # The fixture halves the rate every three epochs, so epoch 5 sits one
    # step below the initial rate.
    assert summary.final_learning_rate == pytest.approx(0.0005)


def test_summarize_training_log_without_log(tmp_path: Any) -> None:
    assert not summarize_training_log(None).available
    assert not summarize_training_log(str(tmp_path / "absent")).available


def test_summarize_explanations(tmp_path: Any) -> None:
    directory = tmp_path / "explanations"
    _write_explanations(str(directory), ["rec_1", "rec_2"])

    summary = summarize_explanations(str(directory))
    assert summary.available
    assert summary.num_explanations == 2
    assert summary.methods == ["integrated_gradients"]
    assert summary.sample_ids == ["rec_1", "rec_2"]


def test_summarize_explanations_without_artifacts(tmp_path: Any) -> None:
    assert not summarize_explanations(str(tmp_path / "absent")).available


def test_summarize_predictions(tmp_path: Any) -> None:
    directory = tmp_path / "predictions"
    _write_predictions(str(directory), count=4)

    summary = summarize_predictions(str(directory))
    assert summary.available
    assert summary.num_predictions == 4
    assert summary.num_failures == 1
    assert summary.statistics["count"] == 4


def test_summarize_predictions_without_artifacts(tmp_path: Any) -> None:
    assert not summarize_predictions(str(tmp_path / "absent")).available


# ---------------------------------------------------------------------------
# Experiment report
# ---------------------------------------------------------------------------


def _build_full_report(tmp_path: Any) -> Any:
    """Builds a run with every artifact type and returns (layout, report)."""
    layout = RunLayout.for_run(str(tmp_path / "runs"), "run_full").create()
    result = _make_result()
    metrics = _write_metrics(layout, result)

    _write_training_log(os.path.join(layout.logs_dir, "csv", "version_0"))
    _write_explanations(layout.explanations_dir, ["rec_1", "rec_2"])
    _write_predictions(layout.predictions_dir)
    figures = generate_all_plots(result, layout.plots_dir, log_dir=layout.logs_dir)

    provenance = collect_provenance(
        run_id="run_full",
        experiment_name="cgcnn_mp_e_form",
        seed=42,
        configs={"experiment": {"seed": 42, "logging": {"level": "INFO"}}},
        config_snapshots=layout.snapshot_configs(
            {"experiment": "configs/experiment.yaml"}
        ),
        now=FIXED_TIME,
    )
    provenance.write(layout.metadata_json)

    reporter = ExperimentReporter(baseline_mae=0.039)
    content = reporter.build(
        layout=layout,
        provenance=provenance,
        metrics=metrics,
        training=summarize_training_log(layout.logs_dir),
        explainability=summarize_explanations(layout.explanations_dir),
        predictions=summarize_predictions(layout.predictions_dir),
        figures=figures,
    )
    return layout, provenance, metrics, figures, content


def test_report_contains_every_required_section(tmp_path: Any) -> None:
    _layout, _prov, _metrics, _figures, content = _build_full_report(tmp_path)

    # PRD §100 required contents.
    for heading in (
        "## 1. Experiment Metadata",
        "## 2. Dataset",
        "## 3. Configuration",
        "## 4. Training Summary",
        "## 5. Evaluation Metrics",
        "## 6. Benchmark Comparison",
        "## 7. Visualizations",
        "## 8. Explainability Summary",
        "## 9. Inference Predictions",
        "## 10. Artifacts",
        "## 11. Conclusions",
        "## 12. Reproducing This Report",
    ):
        assert heading in content


def test_report_records_provenance_and_metrics(tmp_path: Any) -> None:
    _layout, provenance, metrics, _figures, content = _build_full_report(tmp_path)

    assert provenance.generated_at in content
    assert "| Random seed | 42 |" in content
    assert f"{metrics['mae']:.4f}" in content
    assert f"{metrics['rmse']:.4f}" in content
    assert "0.0390" in content  # published baseline
    assert provenance.library_versions["torch"] in content


def test_report_embeds_every_generated_figure(tmp_path: Any) -> None:
    _layout, _prov, _metrics, figures, content = _build_full_report(tmp_path)

    for identifier, caption in FIGURE_TITLES:
        assert identifier in figures
        assert f"![{caption}](plots/{identifier}.png)" in content


def test_report_links_explainability_and_prediction_artifacts(tmp_path: Any) -> None:
    _layout, _prov, _metrics, _figures, content = _build_full_report(tmp_path)

    assert "| Materials explained | 2 |" in content
    assert "`integrated_gradients`" in content
    assert "| Structures scored | 3 |" in content
    assert "explanations/summary.csv" in content
    assert "predictions/predictions.csv" in content


def test_report_is_byte_identical_when_regenerated(tmp_path: Any) -> None:
    # Phase 10 reproducibility criterion: the same run inputs must yield the
    # same report, so nothing may be read from the clock while rendering.
    layout, provenance, metrics, figures, first = _build_full_report(tmp_path)

    reporter = ExperimentReporter(baseline_mae=0.039)
    second = reporter.build(
        layout=layout,
        provenance=provenance,
        metrics=metrics,
        training=summarize_training_log(layout.logs_dir),
        explainability=summarize_explanations(layout.explanations_dir),
        predictions=summarize_predictions(layout.predictions_dir),
        figures=figures,
    )
    assert second == first


def test_report_degrades_when_artifacts_are_missing(tmp_path: Any) -> None:
    layout = RunLayout.for_run(str(tmp_path / "runs"), "run_bare").create()
    provenance = collect_provenance(run_id="run_bare", seed=42, now=FIXED_TIME)

    content = ExperimentReporter(baseline_mae=0.039).build(
        layout=layout,
        provenance=provenance,
        metrics={},
        training=TrainingSummary(),
        explainability=ExplainabilitySummary(),
        predictions=PredictionSummary(),
        figures={},
    )
    assert "No training log was found" in content
    assert "No explainability artifacts" in content
    assert "No inference predictions" in content
    assert "No figures were generated" in content


def test_report_benchmark_discussion_reflects_direction(tmp_path: Any) -> None:
    layout = RunLayout.for_run(str(tmp_path / "runs"), "run_cmp").create()
    provenance = collect_provenance(run_id="run_cmp", now=FIXED_TIME)
    reporter = ExperimentReporter(baseline_mae=0.039)

    better = reporter.build(
        layout=layout, provenance=provenance, metrics={"mae": 0.030, "num_samples": 10}
    )
    worse = reporter.build(
        layout=layout, provenance=provenance, metrics={"mae": 0.050, "num_samples": 10}
    )
    assert "matches or improves" in better
    assert "not a claim of superiority" in better
    assert "reflects this run's training" in worse


def test_experiment_reporter_export_writes_report_md(tmp_path: Any) -> None:
    layout = RunLayout.for_run(str(tmp_path / "runs"), "run_x").create()
    provenance = collect_provenance(run_id="run_x", now=FIXED_TIME)

    path = ExperimentReporter(baseline_mae=0.039).export(
        layout=layout, provenance=provenance, metrics={"mae": 0.04, "num_samples": 5}
    )
    assert path == layout.report_md
    with open(path, "r", encoding="utf-8") as handle:
        assert handle.read().startswith("# Experiment Report — `run_x`")


def test_reporter_summary_accepts_custom_artifact_links(tmp_path: Any) -> None:
    result = _make_result()
    output = tmp_path / "eval"
    Reporter(baseline_mae=0.039).export(
        result,
        str(output),
        artifact_links={"Parity Plot": "plots/parity_plot.png"},
    )
    with open(output / "summary.md", "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "[`plots/parity_plot.png`](plots/parity_plot.png)" in summary
    assert "(parity_plot.png)" not in summary.replace("(plots/parity_plot.png)", "")


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def _patch_evaluator(monkeypatch: pytest.MonkeyPatch, result: EvaluationResult) -> None:
    """Replaces the Evaluator so the CLI runs without the real dataset."""

    class _StubEvaluator:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def run(self) -> EvaluationResult:
            return result

    monkeypatch.setattr("src.cli.report_cli.Evaluator", _StubEvaluator)


def test_report_cli_builds_complete_run(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"weights")
    log_dir = tmp_path / "legacy_logs"
    _write_training_log(str(log_dir))

    experiment_yaml = tmp_path / "experiment.yaml"
    experiment_yaml.write_text(
        "experiment:\n"
        "  name: test_exp\n"
        "  seed: 7\n"
        f"  experiments_dir: {json.dumps(str(tmp_path / 'experiments'))}\n"
        "  runs_dirname: runs\n",
        encoding="utf-8",
    )

    _patch_evaluator(monkeypatch, _make_result())
    monkeypatch.setattr(
        "sys.argv",
        [
            "report_cli.py",
            "--checkpoint",
            str(checkpoint),
            "--experiment-config",
            str(experiment_yaml),
            "--run-id",
            "cli_run",
            "--log-dir",
            str(log_dir),
            "--copy-logs",
            "--copy-checkpoint",
        ],
    )
    report_cli_main()

    layout = RunLayout.for_run(str(tmp_path / "experiments" / "runs"), "cli_run")
    assert os.path.isfile(layout.report_md)
    assert os.path.isfile(layout.metadata_json)
    assert os.path.isfile(layout.metrics_json)
    assert os.path.isfile(layout.predictions_csv)
    assert os.path.isfile(layout.summary_md)
    assert os.path.isfile(os.path.join(layout.checkpoints_dir, "best.ckpt"))
    assert os.path.isfile(os.path.join(layout.logs_dir, "metrics.csv"))
    assert os.path.isfile(os.path.join(layout.plots_dir, "parity_plot.png"))
    assert os.path.isfile(os.path.join(layout.plots_dir, "learning_rate.png"))

    with open(layout.metadata_json, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["seed"] == 7
    assert metadata["run_id"] == "cli_run"
    assert metadata["experiment_name"] == "test_exp"


def test_report_cli_report_only_reproduces_report(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"weights")
    experiment_yaml = tmp_path / "experiment.yaml"
    experiment_yaml.write_text(
        "experiment:\n"
        f"  experiments_dir: {json.dumps(str(tmp_path / 'experiments'))}\n",
        encoding="utf-8",
    )

    _patch_evaluator(monkeypatch, _make_result())
    base_argv = [
        "report_cli.py",
        "--experiment-config",
        str(experiment_yaml),
        "--run-id",
        "repro_run",
    ]
    monkeypatch.setattr("sys.argv", base_argv + ["--checkpoint", str(checkpoint)])
    report_cli_main()

    layout = RunLayout.for_run(str(tmp_path / "experiments" / "runs"), "repro_run")
    with open(layout.report_md, "r", encoding="utf-8") as handle:
        first = handle.read()

    monkeypatch.setattr("sys.argv", base_argv + ["--report-only"])
    report_cli_main()
    with open(layout.report_md, "r", encoding="utf-8") as handle:
        second = handle.read()

    assert second == first


def test_report_cli_requires_checkpoint(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["report_cli.py"])
    with pytest.raises(SystemExit) as excinfo:
        report_cli_main()
    assert excinfo.value.code == 1


def test_report_cli_rejects_missing_checkpoint(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["report_cli.py", "--checkpoint", str(tmp_path / "absent.ckpt")],
    )
    with pytest.raises(SystemExit) as excinfo:
        report_cli_main()
    assert excinfo.value.code == 1


def test_report_cli_refuses_report_only_for_unknown_run(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment_yaml = tmp_path / "experiment.yaml"
    experiment_yaml.write_text(
        "experiment:\n"
        f"  experiments_dir: {json.dumps(str(tmp_path / 'experiments'))}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "report_cli.py",
            "--experiment-config",
            str(experiment_yaml),
            "--run-id",
            "never_created",
            "--report-only",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        report_cli_main()
    assert excinfo.value.code == 1
