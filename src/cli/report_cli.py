"""report_cli.py — CLI entry point for Phase 10 reporting (PRD §121).

Assembles a complete, reproducible experiment run under one directory:
evaluates a checkpoint on the test split, exports metrics and per-sample
predictions, generates every required publication-quality figure, collects
the run's provenance and renders the automated Markdown experiment report.

Usage::

    python src/cli/report_cli.py \\
        --checkpoint experiments/checkpoints/best.ckpt \\
        --log-dir experiments/cgcnn_csv_logs

    # Re-render the report for an existing run without re-evaluating.
    python src/cli/report_cli.py \\
        --run-id cgcnn_mp_e_form_20261010T153000Z \\
        --report-only

Every artifact is written under
``<experiments_dir>/<runs_dirname>/<run_id>/`` as defined by
``configs/experiment.yaml``, so runs never overwrite one another (PRD §18).
"""

import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys
from typing import Any, Dict, Optional

from src.config.dataset_config import DatasetConfig, load_dataset_config
from src.config.exceptions import ConfigurationError
from src.config.experiment_config import ExperimentConfig, load_experiment_config
from src.config.model_config import CGCNNConfig, load_cgcnn_config
from src.evaluation.evaluator import Evaluator
from src.evaluation.experiment_report import (
    ExperimentReporter,
    summarize_explanations,
    summarize_predictions,
    summarize_training_log,
)
from src.evaluation.reporter import Reporter
from src.utils.provenance import collect_provenance, load_provenance
from src.utils.run_layout import RunDirectoryExistsError, RunLayout
from src.visualization.plots import generate_all_plots

logger = logging.getLogger(__name__)

SUMMARY_ARTIFACT_LINKS: Dict[str, str] = {
    "Metrics JSON": "metrics.json",
    "Sample Predictions CSV": "predictions.csv",
    "Experiment Report": "report.md",
    "Run Provenance": "metadata.json",
    "Parity Plot": "plots/parity_plot.png",
    "Residual Distribution Plot": "plots/residual_plot.png",
    "Prediction Histogram": "plots/prediction_histogram.png",
    "Training Loss Curves": "plots/loss_curve.png",
    "Learning Rate Schedule": "plots/learning_rate.png",
}
"""Artifact links for ``summary.md``, pointing into the run's subdirectories."""


def _build_parser() -> argparse.ArgumentParser:
    """Constructs and returns the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the automated experiment report for a run, organising "
            "every artifact under experiments/<runs>/<run_id>/."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to the trained model checkpoint (.ckpt). Required unless "
            "--report-only is used on a run that already has metadata.json."
        ),
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to the dataset YAML configuration.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to the model YAML configuration.",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run identifier. Overrides experiment.run_id; when neither is "
            "set, one is generated as '<name>_<UTC timestamp>'."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Training CSV log file or directory, used for the loss-curve and "
            "learning-rate figures and the training summary. Defaults to the "
            "run's own logs/ directory."
        ),
    )
    parser.add_argument(
        "--copy-logs",
        action="store_true",
        help=(
            "Copy the resolved training log into the run's logs/ directory, "
            "so the run keeps its own copy of the history it reports."
        ),
    )
    parser.add_argument(
        "--copy-checkpoint",
        action="store_true",
        help=(
            "Copy the evaluated checkpoint into the run's checkpoints/ "
            "directory as best.ckpt."
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Re-render report.md from the artifacts already in the run "
            "directory, without re-running evaluation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the run directory if it already exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device override ('cpu', 'cuda', etc.).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader.num_workers from the dataset config.",
    )
    return parser


def _configure_logging(level_name: str) -> None:
    """Configures root logging with the PRD §15 log format.

    Args:
        level_name: Log level name from the experiment configuration.
    """
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _config_payload(
    experiment: ExperimentConfig,
    dataset: DatasetConfig,
    model: CGCNNConfig,
) -> Dict[str, Any]:
    """Returns the configuration values a run used, as plain dictionaries.

    Args:
        experiment: Experiment configuration.
        dataset: Dataset configuration.
        model: Model and optimizer configuration.

    Returns:
        Dict[str, Any]: Serializable snapshot of all three configurations.
    """
    return {
        "experiment": dataclasses.asdict(experiment),
        "dataset": dataclasses.asdict(dataset),
        "model": dataclasses.asdict(model),
    }


def _resolve_log_dir(args: argparse.Namespace, layout: RunLayout) -> Optional[str]:
    """Resolves which training log the report should describe.

    Args:
        args: Parsed command-line arguments.
        layout: The run's directory layout.

    Returns:
        Optional[str]: The log path or directory to read, or ``None`` when no
        log is available.
    """
    if args.log_dir:
        if os.path.exists(args.log_dir):
            return str(args.log_dir)
        logger.warning(
            f"Training log '{args.log_dir}' does not exist; the report will "
            f"omit the training summary and the log-derived figures."
        )
        return None
    if os.path.isdir(layout.logs_dir):
        return layout.logs_dir
    return None


def _copy_training_log(log_dir: str, layout: RunLayout) -> Optional[str]:
    """Copies a resolved training log into the run's logs directory.

    Args:
        log_dir: The resolved log file or directory.
        layout: The run's directory layout.

    Returns:
        Optional[str]: The destination path, or ``None`` when nothing was
        copied.
    """
    os.makedirs(layout.logs_dir, exist_ok=True)
    source = log_dir
    if os.path.isdir(log_dir):
        matches = [
            os.path.join(root, "metrics.csv")
            for root, _dirs, files in os.walk(log_dir)
            if "metrics.csv" in files
        ]
        if not matches:
            return None
        source = max(matches, key=os.path.getmtime)

    destination = os.path.join(layout.logs_dir, "metrics.csv")
    if os.path.abspath(source) == os.path.abspath(destination):
        return destination
    shutil.copyfile(source, destination)
    logger.info(f"Copied training log '{source}' to '{destination}'.")
    return destination


def _read_metrics(layout: RunLayout) -> Dict[str, Any]:
    """Reads the run's exported metrics document.

    Args:
        layout: The run's directory layout.

    Returns:
        Dict[str, Any]: The parsed metrics, or an empty mapping when absent.
    """
    if not os.path.isfile(layout.metrics_json):
        return {}
    try:
        with open(layout.metrics_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read '{layout.metrics_json}': {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _discover_figures(layout: RunLayout) -> Dict[str, str]:
    """Finds the figures already present in a run's plots directory.

    Args:
        layout: The run's directory layout.

    Returns:
        Dict[str, str]: Mapping of figure identifier (the file stem) to path.
    """
    if not os.path.isdir(layout.plots_dir):
        return {}
    return {
        os.path.splitext(name)[0]: os.path.join(layout.plots_dir, name)
        for name in sorted(os.listdir(layout.plots_dir))
        if name.lower().endswith(".png")
    }


def _run_evaluation(
    args: argparse.Namespace,
    layout: RunLayout,
    experiment: ExperimentConfig,
    log_dir: Optional[str],
) -> Dict[str, Any]:
    """Evaluates the checkpoint and writes the run's evaluation artifacts.

    Args:
        args: Parsed command-line arguments.
        layout: The run's directory layout.
        experiment: Experiment configuration.
        log_dir: Resolved training log, used for the log-derived figures.

    Returns:
        Dict[str, Any]: The metrics document written to ``metrics.json``.
    """
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        model_config_path=args.model_config,
        device=args.device,
        num_workers=args.num_workers,
        seed=experiment.seed,
    )

    logger.info("Executing evaluation pipeline...")
    result = evaluator.run()

    reporter = Reporter(baseline_mae=experiment.baseline_mae_ev_atom)
    reporter.export(result, layout.root, artifact_links=SUMMARY_ARTIFACT_LINKS)
    logger.info(f"Exported evaluation artifacts to '{layout.root}'.")

    logger.info("Generating publication-quality figures...")
    generate_all_plots(result=result, output_dir=layout.plots_dir, log_dir=log_dir)

    return _read_metrics(layout)


def main() -> None:
    """Main CLI entry point for Phase 10 reporting."""
    args = _build_parser().parse_args()

    try:
        experiment = load_experiment_config(args.experiment_config)
    except ConfigurationError as exc:
        logging.basicConfig(level=logging.INFO)
        logger.error(str(exc))
        sys.exit(1)

    _configure_logging(experiment.logging.level)

    if not args.report_only and not args.checkpoint:
        logger.error("--checkpoint is required unless --report-only is used.")
        sys.exit(1)
    if args.checkpoint and not os.path.isfile(args.checkpoint):
        logger.error(f"Checkpoint file not found at: {args.checkpoint}")
        sys.exit(1)

    run_id = experiment.resolve_run_id(args.run_id)
    layout = RunLayout.for_run(experiment.runs_root, run_id)

    if args.report_only and not os.path.isdir(layout.root):
        logger.error(
            f"--report-only was given but run directory '{layout.root}' does "
            f"not exist."
        )
        sys.exit(1)

    try:
        layout.create(exist_ok=args.report_only, overwrite=args.overwrite)
    except RunDirectoryExistsError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {layout.root}")

    dataset_config = load_dataset_config(args.dataset_config)
    model_config = load_cgcnn_config(args.model_config)

    snapshots = layout.snapshot_configs(
        {
            "dataset": args.dataset_config,
            "model": args.model_config,
            "experiment": args.experiment_config,
        }
    )

    log_dir = _resolve_log_dir(args, layout)
    if log_dir and args.copy_logs:
        copied = _copy_training_log(log_dir, layout)
        if copied:
            log_dir = layout.logs_dir

    if args.report_only:
        metrics = _read_metrics(layout)
        figures = _discover_figures(layout)
        if not metrics:
            logger.warning(
                f"No metrics.json in '{layout.root}'; the report will omit "
                f"the evaluation metrics."
            )
    else:
        metrics = _run_evaluation(args, layout, experiment, log_dir)
        figures = _discover_figures(layout)

    checkpoint_path = args.checkpoint
    if checkpoint_path and args.copy_checkpoint:
        os.makedirs(layout.checkpoints_dir, exist_ok=True)
        destination = os.path.join(layout.checkpoints_dir, "best.ckpt")
        if os.path.abspath(checkpoint_path) != os.path.abspath(destination):
            shutil.copyfile(checkpoint_path, destination)
            logger.info(f"Copied checkpoint to '{destination}'.")

    # A report-only rerun reuses the recorded provenance so the regenerated
    # report is byte-identical to the original (Phase 10 reproducibility).
    provenance = None
    if args.report_only and os.path.isfile(layout.metadata_json):
        try:
            provenance = load_provenance(layout.metadata_json)
            logger.info(f"Reusing recorded provenance from '{layout.metadata_json}'.")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                f"Could not reuse '{layout.metadata_json}' ({exc}); collecting "
                f"provenance afresh."
            )

    if provenance is None:
        provenance = collect_provenance(
            run_id=run_id,
            experiment_name=experiment.name,
            seed=experiment.seed,
            checkpoint_path=checkpoint_path,
            configs=_config_payload(experiment, dataset_config, model_config),
            config_snapshots=snapshots,
            raw_data_path=dataset_config.raw_data_path,
            metadata_dir=dataset_config.metadata_dir,
            processed_data_dir=dataset_config.processed_data_dir,
            target_column=dataset_config.target_column,
        )
        provenance.write(layout.metadata_json)
        logger.info(f"Wrote run provenance to '{layout.metadata_json}'.")

    experiment_reporter = ExperimentReporter(
        baseline_mae=experiment.baseline_mae_ev_atom
    )
    report_path = experiment_reporter.export(
        layout=layout,
        provenance=provenance,
        metrics=metrics,
        training=summarize_training_log(log_dir),
        explainability=summarize_explanations(layout.explanations_dir),
        predictions=summarize_predictions(layout.predictions_dir),
        figures=figures,
    )

    print("\n" + "=" * 68)
    print("                   CGCNN EXPERIMENT REPORT")
    print("=" * 68)
    print(f" Run ID        : {run_id}")
    print(f" Run directory : {layout.root}")
    if metrics:
        print(f" Test samples  : {metrics.get('num_samples', '-')}")
        print(f" MAE (eV/atom) : {metrics.get('mae', float('nan')):.4f}")
        print(f" RMSE (eV/atom): {metrics.get('rmse', float('nan')):.4f}")
        print(f" R²            : {metrics.get('r2', float('nan')):.4f}")
    print("-" * 68)
    print(f" Report        : {report_path}")
    print(f" Provenance    : {layout.metadata_json}")
    print(f" Figures       : {len(figures)} in {layout.plots_dir}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
