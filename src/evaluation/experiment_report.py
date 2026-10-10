"""Automated experiment report generation (PRD §100).

Assembles one Markdown document describing a complete run: provenance,
dataset version, configuration, training summary, test-set metrics, the
published-baseline comparison, every generated figure and the explainability
and prediction artifacts produced for the run.

The report is a pure function of its inputs. Nothing is read from the clock
or the environment while rendering, so regenerating a report from the same
run inputs produces byte-identical output — the reproducibility criterion for
Phase 10. Collecting those inputs (which does touch the filesystem) is the
job of the ``summarize_*`` helpers and :mod:`src.utils.provenance`.

This module never generates figures; it only links the ones it is given, so
the evaluation package keeps no dependency on the visualization package
(PRD §11).
"""

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.utils.provenance import RunProvenance
from src.utils.run_layout import RunLayout

logger = logging.getLogger(__name__)

FIGURE_TITLES: Tuple[Tuple[str, str], ...] = (
    ("loss_curve", "Training and Validation Loss"),
    ("learning_rate", "Learning Rate Schedule"),
    ("parity_plot", "Prediction Parity"),
    ("residual_plot", "Residual Distribution"),
    ("prediction_histogram", "Prediction Histogram"),
)
"""Figure identifiers and captions, in the order they appear in the report.

The order matches PRD §95's list of required visualizations, so a reader
reviewing two runs sees the same figures in the same sequence.
"""

_NA = "_not available_"


def _fmt(value: Optional[float], digits: int = 4, signed: bool = False) -> str:
    """Formats an optional float for a Markdown table cell."""
    if value is None:
        return "–"
    if signed:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def _fmt_int(value: Optional[int]) -> str:
    """Formats an optional integer with thousands separators."""
    return "–" if value is None else f"{value:,}"


def _fmt_str(value: Optional[Any]) -> str:
    """Formats an optional scalar as a Markdown cell, dash when absent."""
    if value is None or value == "":
        return "–"
    return str(value)


@dataclass
class TrainingSummary:
    """Training history of the run, read back from its CSV log.

    Attributes:
        log_path: CSV log the summary was derived from.
        epochs_completed: Number of epochs present in the log.
        best_val_loss: Lowest recorded validation loss.
        best_val_loss_epoch: Epoch at which that loss occurred.
        best_val_mae: Lowest recorded validation MAE in eV/atom.
        final_train_loss: Training loss at the last logged epoch.
        final_val_loss: Validation loss at the last logged epoch.
        initial_learning_rate: First recorded learning rate, if logged.
        final_learning_rate: Last recorded learning rate, if logged.
    """

    log_path: Optional[str] = None
    epochs_completed: Optional[int] = None
    best_val_loss: Optional[float] = None
    best_val_loss_epoch: Optional[int] = None
    best_val_mae: Optional[float] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    initial_learning_rate: Optional[float] = None
    final_learning_rate: Optional[float] = None

    @property
    def available(self) -> bool:
        """Whether any training history was recovered."""
        return self.epochs_completed is not None


@dataclass
class ExplainabilitySummary:
    """Explainability artifacts produced for the run (PRD §86, §87).

    Attributes:
        summary_csv: Path of the aggregated explanation summary, if present.
        num_explanations: Number of explained materials.
        methods: Attribution methods represented in the artifacts.
        sample_ids: Identifiers of the explained materials.
    """

    summary_csv: Optional[str] = None
    num_explanations: int = 0
    methods: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        """Whether any explanation artifacts were found."""
        return self.num_explanations > 0


@dataclass
class PredictionSummary:
    """Inference predictions produced for the run (PRD §91, §93).

    Attributes:
        predictions_csv: Path of the exported predictions CSV, if present.
        predictions_json: Path of the exported predictions JSON, if present.
        num_predictions: Number of structures scored.
        num_failures: Structures that were skipped.
        statistics: Aggregate statistics recorded by the exporter.
    """

    predictions_csv: Optional[str] = None
    predictions_json: Optional[str] = None
    num_predictions: int = 0
    num_failures: int = 0
    statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """Whether any prediction artifacts were found."""
        return self.num_predictions > 0 or self.predictions_csv is not None


def summarize_training_log(log_path_or_dir: Optional[str]) -> TrainingSummary:
    """Summarizes a run's training history from its CSV log.

    Args:
        log_path_or_dir: A ``metrics.csv`` path, or a directory containing one
            at any depth. ``None`` yields an empty summary.

    Returns:
        TrainingSummary: The recovered history, empty when no usable log
        exists. A missing log is not an error: a checkpoint can be evaluated
        and reported without the logs of the run that produced it.
    """
    summary = TrainingSummary()
    if not log_path_or_dir:
        return summary

    csv_path: Optional[str] = None
    if os.path.isfile(log_path_or_dir):
        csv_path = log_path_or_dir
    elif os.path.isdir(log_path_or_dir):
        matches = [
            os.path.join(root, "metrics.csv")
            for root, _dirs, files in os.walk(log_path_or_dir)
            if "metrics.csv" in files
        ]
        if matches:
            csv_path = max(matches, key=os.path.getmtime)

    if csv_path is None:
        return summary
    summary.log_path = csv_path

    try:
        frame = pd.read_csv(csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning(f"Could not read training log '{csv_path}': {exc}")
        return summary

    if "epoch" not in frame.columns or frame.empty:
        return summary

    grouped = frame.groupby("epoch").mean(numeric_only=True)
    if grouped.empty:
        return summary

    summary.epochs_completed = int(len(grouped))

    if "val_loss" in grouped.columns:
        val_loss = grouped["val_loss"].dropna()
        if not val_loss.empty:
            summary.best_val_loss = float(val_loss.min())
            summary.best_val_loss_epoch = _as_int(val_loss.idxmin())
            summary.final_val_loss = float(val_loss.iloc[-1])

    for column in ("val_MAE", "val_mae"):
        if column in grouped.columns:
            val_mae = grouped[column].dropna()
            if not val_mae.empty:
                summary.best_val_mae = float(val_mae.min())
                break

    if "train_loss" in grouped.columns:
        train_loss = grouped["train_loss"].dropna()
        if not train_loss.empty:
            summary.final_train_loss = float(train_loss.iloc[-1])

    lr_column = next(
        (
            name
            for name in grouped.columns
            if str(name).startswith("lr") and not grouped[name].isna().all()
        ),
        None,
    )
    if lr_column is not None:
        learning_rates = grouped[lr_column].dropna()
        summary.initial_learning_rate = float(learning_rates.iloc[0])
        summary.final_learning_rate = float(learning_rates.iloc[-1])

    return summary


def summarize_explanations(explanations_dir: Optional[str]) -> ExplainabilitySummary:
    """Summarizes the explainability artifacts in a run directory.

    Args:
        explanations_dir: Directory written by the explainability exporter.

    Returns:
        ExplainabilitySummary: The recovered summary, empty when the
        directory is missing or holds no explanations.
    """
    summary = ExplainabilitySummary()
    if not explanations_dir or not os.path.isdir(explanations_dir):
        return summary

    summary.sample_ids = sorted(
        entry
        for entry in os.listdir(explanations_dir)
        if os.path.isfile(os.path.join(explanations_dir, entry, "explanation.json"))
    )
    summary.num_explanations = len(summary.sample_ids)

    summary_csv = os.path.join(explanations_dir, "summary.csv")
    if not os.path.isfile(summary_csv):
        return summary
    summary.summary_csv = summary_csv

    try:
        with open(summary_csv, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        logger.warning(f"Could not read '{summary_csv}': {exc}")
        return summary

    if rows and not summary.sample_ids:
        summary.sample_ids = sorted(
            str(row.get("sample_id", "")) for row in rows if row.get("sample_id")
        )
        summary.num_explanations = len(summary.sample_ids)

    summary.methods = sorted({str(row["method"]) for row in rows if row.get("method")})
    return summary


def summarize_predictions(predictions_dir: Optional[str]) -> PredictionSummary:
    """Summarizes the inference predictions in a run directory.

    Args:
        predictions_dir: Directory written by the prediction exporter.

    Returns:
        PredictionSummary: The recovered summary, empty when the directory is
        missing or holds no predictions.
    """
    summary = PredictionSummary()
    if not predictions_dir or not os.path.isdir(predictions_dir):
        return summary

    csv_path = os.path.join(predictions_dir, "predictions.csv")
    json_path = os.path.join(predictions_dir, "predictions.json")

    if os.path.isfile(csv_path):
        summary.predictions_csv = csv_path
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as handle:
                summary.num_predictions = sum(1 for _ in csv.DictReader(handle))
        except OSError as exc:
            logger.warning(f"Could not read '{csv_path}': {exc}")

    if os.path.isfile(json_path):
        summary.predictions_json = json_path
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read '{json_path}': {exc}")
            return summary
        if isinstance(payload, dict):
            statistics = payload.get("summary")
            if isinstance(statistics, dict):
                summary.statistics = statistics
            failures = payload.get("failures")
            if isinstance(failures, list):
                summary.num_failures = len(failures)

    return summary


class ExperimentReporter:
    """Renders the automated Markdown experiment report (PRD §100).

    The report is the primary artifact for reviewing a run, so it gathers
    every fact needed to judge and reproduce that run into one document.
    """

    def __init__(self, baseline_mae: float) -> None:
        """Initializes the reporter.

        Args:
            baseline_mae: Published CGCNN baseline MAE in eV/atom, used for
                the benchmark comparison (PRD §99).
        """
        self.baseline_mae = baseline_mae

    def build(
        self,
        layout: RunLayout,
        provenance: RunProvenance,
        metrics: Dict[str, Any],
        training: Optional[TrainingSummary] = None,
        explainability: Optional[ExplainabilitySummary] = None,
        predictions: Optional[PredictionSummary] = None,
        figures: Optional[Dict[str, str]] = None,
    ) -> str:
        """Renders the report as a Markdown string.

        Args:
            layout: The run's directory layout, used to express every link
                relative to the run root so the report stays portable.
            provenance: The run's provenance record.
            metrics: Test-set metrics as written to ``metrics.json``.
            training: Training history summary, if available.
            explainability: Explainability artifact summary, if available.
            predictions: Inference artifact summary, if available.
            figures: Mapping of figure identifier to generated file path.

        Returns:
            str: The complete Markdown report.
        """
        sections = [
            self._header(provenance),
            self._metadata_section(provenance),
            self._dataset_section(provenance),
            self._configuration_section(layout, provenance),
            self._training_section(layout, training or TrainingSummary()),
            self._metrics_section(metrics),
            self._benchmark_section(metrics),
            self._figures_section(layout, figures or {}),
            self._explainability_section(
                layout, explainability or ExplainabilitySummary()
            ),
            self._predictions_section(layout, predictions or PredictionSummary()),
            self._artifacts_section(layout),
            self._conclusions_section(metrics, training or TrainingSummary()),
            self._reproduction_section(layout, provenance),
        ]
        return "\n\n---\n\n".join(section.strip() for section in sections) + "\n"

    def export(
        self,
        layout: RunLayout,
        provenance: RunProvenance,
        metrics: Dict[str, Any],
        training: Optional[TrainingSummary] = None,
        explainability: Optional[ExplainabilitySummary] = None,
        predictions: Optional[PredictionSummary] = None,
        figures: Optional[Dict[str, str]] = None,
    ) -> str:
        """Renders the report and writes it to the run's ``report.md``.

        Args:
            layout: The run's directory layout.
            provenance: The run's provenance record.
            metrics: Test-set metrics as written to ``metrics.json``.
            training: Training history summary, if available.
            explainability: Explainability artifact summary, if available.
            predictions: Inference artifact summary, if available.
            figures: Mapping of figure identifier to generated file path.

        Returns:
            str: The path of the written report.
        """
        content = self.build(
            layout=layout,
            provenance=provenance,
            metrics=metrics,
            training=training,
            explainability=explainability,
            predictions=predictions,
            figures=figures,
        )
        os.makedirs(layout.root, exist_ok=True)
        with open(layout.report_md, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        logger.info(f"Wrote experiment report to '{layout.report_md}'.")
        return layout.report_md

    # -- Sections ---------------------------------------------------------

    def _header(self, provenance: RunProvenance) -> str:
        """Renders the report title."""
        return (
            f"# Experiment Report — `{provenance.run_id}`\n\n"
            f"Automated report for a single CGCNN formation-energy run "
            f"(PRD §100). Every path below is relative to this run directory."
        )

    def _metadata_section(self, provenance: RunProvenance) -> str:
        """Renders the experiment metadata and provenance table (PRD §17)."""
        git = provenance.git
        hardware = provenance.hardware
        dirty = "–" if git.dirty is None else ("yes" if git.dirty else "no")
        gpus = ", ".join(hardware.gpu_names) if hardware.gpu_names else "none"

        rows = [
            ("Run ID", f"`{provenance.run_id}`"),
            ("Experiment name", _fmt_str(provenance.experiment_name)),
            ("Generated at (UTC)", _fmt_str(provenance.generated_at)),
            ("Random seed", _fmt_str(provenance.seed)),
            ("Git commit", f"`{git.commit}`" if git.commit else "–"),
            ("Git branch", _fmt_str(git.branch)),
            ("Uncommitted changes", dirty),
            ("Model version", _fmt_str(provenance.model_version)),
            (
                "Checkpoint",
                (
                    f"`{provenance.checkpoint_path}`"
                    if provenance.checkpoint_path
                    else "–"
                ),
            ),
            (
                "Checkpoint SHA-256",
                (
                    f"`{provenance.checkpoint_sha256}`"
                    if provenance.checkpoint_sha256
                    else "–"
                ),
            ),
            ("Platform", _fmt_str(hardware.platform)),
            ("Processor", _fmt_str(hardware.processor or hardware.machine)),
            ("Logical CPUs", _fmt_str(hardware.cpu_count)),
            ("CUDA available", "yes" if hardware.cuda_available else "no"),
            ("CUDA version", _fmt_str(hardware.cuda_version)),
            ("cuDNN version", _fmt_str(hardware.cudnn_version)),
            ("GPU(s)", gpus),
            ("Python", _fmt_str(hardware.python_version)),
        ]
        table = _table(("Field", "Value"), rows)

        versions = [
            (f"`{name}`", _fmt_str(version))
            for name, version in sorted(provenance.library_versions.items())
        ]
        library_table = _table(("Library", "Version"), versions)

        return (
            "## 1. Experiment Metadata\n\n"
            f"{table}\n\n"
            "### Library Versions\n\n"
            f"{library_table}"
        )

    def _dataset_section(self, provenance: RunProvenance) -> str:
        """Renders the dataset fingerprint and record counts."""
        dataset = provenance.dataset
        rows = [
            (
                "Dataset fingerprint (SHA-256)",
                f"`{dataset.fingerprint}`" if dataset.fingerprint else "–",
            ),
            ("Raw dataset", _fmt_str(dataset.raw_data_path)),
            ("Processed graphs", _fmt_str(dataset.processed_data_dir)),
            ("Target column", _fmt_str(dataset.target_column)),
            ("Total records", _fmt_int(dataset.total_records)),
            ("Valid records", _fmt_int(dataset.valid_records)),
            ("Graphs built", _fmt_int(dataset.successful_graphs)),
            ("Graphs failed", _fmt_int(dataset.failed_graphs)),
            ("Shards", _fmt_int(dataset.shards_created)),
        ]
        section = "## 2. Dataset\n\n" + _table(("Field", "Value"), rows)

        if dataset.graph_config:
            graph_rows = [
                (f"`{key}`", _fmt_str(value))
                for key, value in sorted(dataset.graph_config.items())
            ]
            section += "\n\n### Graph Construction\n\n" + _table(
                ("Setting", "Value"), graph_rows
            )
        return section

    def _configuration_section(
        self, layout: RunLayout, provenance: RunProvenance
    ) -> str:
        """Renders the configuration the run used, plus snapshot links."""
        section = "## 3. Configuration\n\n"

        if not provenance.configs:
            return section + _NA

        for name in sorted(provenance.configs):
            values = provenance.configs[name]
            section += f"### `{name}`\n\n"
            rows = [
                (f"`{key}`", _fmt_str(value)) for key, value in _flatten(values).items()
            ]
            section += _table(("Setting", "Value"), rows) + "\n\n"

        snapshots = [
            (name, layout.relative(path))
            for name, path in sorted(provenance.config_snapshots.items())
        ]
        if snapshots:
            section += "### Snapshots\n\n" + "\n".join(
                f"- `{name}`: [`{path}`]({path})" for name, path in snapshots
            )
        return section

    def _training_section(self, layout: RunLayout, training: TrainingSummary) -> str:
        """Renders the training summary recovered from the run's logs."""
        if not training.available:
            return (
                "## 4. Training Summary\n\n"
                "No training log was found for this run, so the training "
                "history could not be summarized. Evaluation metrics below "
                "are unaffected."
            )

        rows = [
            ("Epochs completed", _fmt_int(training.epochs_completed)),
            ("Best validation loss (MSE)", _fmt(training.best_val_loss, 6)),
            ("Best epoch", _fmt_str(training.best_val_loss_epoch)),
            (
                "Best validation MAE (eV/atom)",
                _fmt(training.best_val_mae),
            ),
            ("Final training loss (MSE)", _fmt(training.final_train_loss, 6)),
            ("Final validation loss (MSE)", _fmt(training.final_val_loss, 6)),
            (
                "Initial learning rate",
                (
                    "–"
                    if training.initial_learning_rate is None
                    else f"{training.initial_learning_rate:.3e}"
                ),
            ),
            (
                "Final learning rate",
                (
                    "–"
                    if training.final_learning_rate is None
                    else f"{training.final_learning_rate:.3e}"
                ),
            ),
            ("Training log", _fmt_str(layout.relative(training.log_path))),
        ]
        return "## 4. Training Summary\n\n" + _table(("Field", "Value"), rows)

    def _metrics_section(self, metrics: Dict[str, Any]) -> str:
        """Renders the test-set metrics table (PRD §78)."""
        rows = [
            (
                "Test samples",
                _fmt_int(_as_int(metrics.get("num_samples"))),
                "–",
                "Graphs in the held-out test split",
            ),
            (
                "MAE",
                _fmt(_as_float(metrics.get("mae"))),
                "eV/atom",
                "Mean absolute error (primary benchmark metric)",
            ),
            (
                "RMSE",
                _fmt(_as_float(metrics.get("rmse"))),
                "eV/atom",
                "Root mean square error",
            ),
            (
                "R²",
                _fmt(_as_float(metrics.get("r2"))),
                "–",
                "Coefficient of determination",
            ),
            (
                "Mean bias",
                _fmt(_as_float(metrics.get("bias")), signed=True),
                "eV/atom",
                "Mean signed error (prediction − truth)",
            ),
        ]
        return "## 5. Evaluation Metrics\n\n" + _table(
            ("Metric", "Value", "Unit", "Description"), rows
        )

    def _benchmark_section(self, metrics: Dict[str, Any]) -> str:
        """Renders the published-baseline comparison (PRD §99)."""
        comparison = metrics.get("benchmark_comparison")
        mae = _as_float(metrics.get("mae"))
        baseline = self.baseline_mae
        dataset_name = "matbench_mp_e_form"

        if isinstance(comparison, dict):
            baseline = (
                _as_float(comparison.get("baseline_mae_ev_atom")) or self.baseline_mae
            )
            dataset_name = str(comparison.get("benchmark_dataset", dataset_name))

        if mae is None:
            return "## 6. Benchmark Comparison\n\n" + _NA

        difference = mae - baseline
        relative = (difference / baseline * 100.0) if baseline else 0.0

        rows = [
            ("Published CGCNN baseline", _fmt(baseline), "–", "–"),
            (
                "This run (test split)",
                _fmt(mae),
                _fmt(difference, signed=True),
                f"{relative:+.2f}%",
            ),
        ]
        table = _table(
            ("Model", "MAE (eV/atom)", "Difference (eV/atom)", "Relative"), rows
        )

        if difference <= 0:
            discussion = (
                f"This run's test MAE of {mae:.4f} eV/atom matches or improves "
                f"on the published CGCNN baseline of {baseline:.4f} eV/atom on "
                f"`{dataset_name}`, by {abs(relative):.2f}%. The comparison "
                f"verifies that the pipeline reproduces known behaviour; it is "
                f"not a claim of superiority, which would require a "
                f"statistically controlled comparison on the official split."
            )
        else:
            discussion = (
                f"This run's test MAE of {mae:.4f} eV/atom is "
                f"{difference:+.4f} eV/atom ({relative:+.2f}%) from the "
                f"published CGCNN baseline of {baseline:.4f} eV/atom on "
                f"`{dataset_name}`. The gap reflects this run's training "
                f"budget, hyperparameters and split partition rather than a "
                f"defect in the pipeline."
            )

        return (
            "## 6. Benchmark Comparison\n\n"
            f"{table}\n\n"
            f"### Discussion\n\n{discussion}"
        )

    def _figures_section(self, layout: RunLayout, figures: Dict[str, str]) -> str:
        """Renders every generated figure as an embedded image (PRD §95)."""
        section = "## 7. Visualizations\n\n"
        rendered = 0

        for identifier, caption in FIGURE_TITLES:
            path = figures.get(identifier)
            if not path:
                continue
            relative = layout.relative(path)
            section += (
                f"### {caption}\n\n"
                f"![{caption}]({relative})\n\n"
                f"Source: [`{relative}`]({relative})\n\n"
            )
            rendered += 1

        extra = sorted(set(figures) - {key for key, _ in FIGURE_TITLES})
        for identifier in extra:
            relative = layout.relative(figures[identifier])
            section += (
                f"### {identifier}\n\n"
                f"![{identifier}]({relative})\n\n"
                f"Source: [`{relative}`]({relative})\n\n"
            )
            rendered += 1

        if rendered == 0:
            section += "No figures were generated for this run."
        return section

    def _explainability_section(
        self, layout: RunLayout, explainability: ExplainabilitySummary
    ) -> str:
        """Renders the explainability artifact summary (PRD §86, §87)."""
        section = "## 8. Explainability Summary\n\n"
        if not explainability.available:
            return section + (
                "No explainability artifacts were produced for this run. Run "
                "`python src/cli/explain_cli.py --output-dir "
                "<run>/explanations` to add them."
            )

        methods = ", ".join(f"`{m}`" for m in explainability.methods) or "–"
        rows = [
            ("Materials explained", _fmt_int(explainability.num_explanations)),
            ("Attribution methods", methods),
            (
                "Summary CSV",
                _fmt_str(layout.relative(explainability.summary_csv)),
            ),
        ]
        section += _table(("Field", "Value"), rows)

        preview = explainability.sample_ids[:10]
        if preview:
            listed = ", ".join(f"`{sample}`" for sample in preview)
            remainder = explainability.num_explanations - len(preview)
            suffix = f" (+{remainder} more)" if remainder > 0 else ""
            section += f"\n\nExplained samples: {listed}{suffix}"

        section += (
            "\n\nAttribution outputs are interpretations of the learned model, "
            "not physical validation of the underlying chemistry "
            "(PRD §82, §101)."
        )
        return section

    def _predictions_section(
        self, layout: RunLayout, predictions: PredictionSummary
    ) -> str:
        """Renders the inference artifact summary (PRD §91, §93)."""
        section = "## 9. Inference Predictions\n\n"
        if not predictions.available:
            return section + (
                "No inference predictions were exported for this run. Run "
                "`python src/cli/predict_cli.py predict-batch --output-dir "
                "<run>/predictions` to add them."
            )

        rows = [
            ("Structures scored", _fmt_int(predictions.num_predictions)),
            ("Structures skipped", _fmt_int(predictions.num_failures)),
            ("Predictions CSV", _fmt_str(layout.relative(predictions.predictions_csv))),
            (
                "Predictions JSON",
                _fmt_str(layout.relative(predictions.predictions_json)),
            ),
        ]
        section += _table(("Field", "Value"), rows)

        if predictions.statistics:
            stat_rows = [
                (f"`{key}`", _fmt_str(value))
                for key, value in sorted(predictions.statistics.items())
            ]
            section += "\n\n### Aggregate Statistics\n\n" + _table(
                ("Statistic", "Value"), stat_rows
            )
        return section

    def _artifacts_section(self, layout: RunLayout) -> str:
        """Renders the canonical artifact layout of the run (PRD §19, §98)."""
        entries = [
            ("Run provenance", layout.relative(layout.metadata_json)),
            ("Test-set metrics", layout.relative(layout.metrics_json)),
            ("Per-sample predictions", layout.relative(layout.predictions_csv)),
            ("Evaluation summary", layout.relative(layout.summary_md)),
            ("Configuration snapshots", _as_dir(layout.relative(layout.config_dir))),
            ("Checkpoints", _as_dir(layout.relative(layout.checkpoints_dir))),
            ("Logs", _as_dir(layout.relative(layout.logs_dir))),
            ("Figures", _as_dir(layout.relative(layout.plots_dir))),
            ("Explanations", _as_dir(layout.relative(layout.explanations_dir))),
            ("Inference predictions", _as_dir(layout.relative(layout.predictions_dir))),
        ]
        rows = [
            (name, f"[`{path}`]({path})" if path else "–") for name, path in entries
        ]
        return "## 10. Artifacts\n\n" + _table(("Artifact", "Path"), rows)

    def _conclusions_section(
        self, metrics: Dict[str, Any], training: TrainingSummary
    ) -> str:
        """Renders the conclusions drawn from this run's numbers."""
        mae = _as_float(metrics.get("mae"))
        r2 = _as_float(metrics.get("r2"))
        bias = _as_float(metrics.get("bias"))
        samples = _as_int(metrics.get("num_samples"))

        if mae is None:
            return "## 11. Conclusions\n\n" + _NA

        lines = [
            f"- The model reaches a test MAE of **{mae:.4f} eV/atom** on "
            f"{_fmt_int(samples)} held-out graphs, against a published CGCNN "
            f"baseline of {self.baseline_mae:.4f} eV/atom.",
        ]
        if r2 is not None:
            lines.append(
                f"- An R² of **{r2:.4f}** indicates the fraction of target "
                f"variance the model explains on the test split."
            )
        if bias is not None:
            direction = "over-predicts" if bias > 0 else "under-predicts"
            lines.append(
                f"- The mean bias of **{bias:+.4f} eV/atom** shows the model "
                f"slightly {direction} formation energy on average; a bias far "
                f"from zero would point at a systematic offset rather than "
                f"random error."
            )
        if training.available and training.best_val_loss is not None:
            lines.append(
                f"- Training ran for {_fmt_int(training.epochs_completed)} "
                f"epochs, with the best validation loss of "
                f"{training.best_val_loss:.6f} at epoch "
                f"{_fmt_str(training.best_val_loss_epoch)}."
            )
        lines.append(
            "- Explainability outputs, where present, are interpretations of "
            "the learned model and are not evidence of physical causation "
            "(PRD §101)."
        )
        return "## 11. Conclusions\n\n" + "\n".join(lines)

    def _reproduction_section(
        self, layout: RunLayout, provenance: RunProvenance
    ) -> str:
        """Renders the exact command that regenerates this report."""
        checkpoint = provenance.checkpoint_path or "<checkpoint>.ckpt"
        command = (
            "python src/cli/report_cli.py \\\n"
            f"    --checkpoint {checkpoint} \\\n"
            f"    --run-id {provenance.run_id}"
        )
        config_note = (
            "The configuration snapshots under "
            f"`{layout.relative(layout.config_dir)}/` record the settings this "
            "run used; pass them with `--dataset-config`, `--model-config` and "
            "`--experiment-config` to reproduce the run after `configs/` has "
            "changed."
        )
        return (
            "## 12. Reproducing This Report\n\n"
            f"```bash\n{command}\n```\n\n{config_note}"
        )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Renders a Markdown table.

    Args:
        headers: Column headers.
        rows: Row cells, already formatted as strings.

    Returns:
        str: The rendered table, or a placeholder when there are no rows.
    """
    if not rows:
        return _NA
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(cell) for cell in row) + " |" for row in rows
    )
    return f"{header_line}\n{divider}\n{body}"


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flattens a nested configuration mapping into dotted keys.

    Args:
        value: A mapping, or a scalar to report under ``prefix``.
        prefix: Dotted key prefix accumulated so far.

    Returns:
        Dict[str, Any]: Mapping of dotted key to scalar value.
    """
    if not isinstance(value, dict):
        return {prefix or "value": value}

    flattened: Dict[str, Any] = {}
    for key in sorted(value, key=str):
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(value[key], child_prefix))
    return flattened


def _as_dir(path: Optional[str]) -> Optional[str]:
    """Appends a trailing slash so a directory link reads as a directory."""
    return None if path is None else f"{path}/"


def _as_float(value: Any) -> Optional[float]:
    """Coerces a metric value to float, returning ``None`` when impossible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    """Coerces a value to int, returning ``None`` when impossible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
