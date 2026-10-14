"""Unified CLI entry point for the formation energy prediction pipeline.

Provides a single ``project <command>`` interface wrapping every subsystem
CLI (PRD §121).  Each subcommand delegates to the existing ``*_cli.py``
module so flags and tested behaviour are preserved.

Usage::

    python -m src.cli.main_cli dataset validate
    python -m src.cli.main_cli train --max-epochs 50
    python -m src.cli.main_cli evaluate --checkpoint best.ckpt
    python -m src.cli.main_cli explain  --checkpoint best.ckpt --output-dir out
    python -m src.cli.main_cli predict  --checkpoint best.ckpt --input s.cif
    python -m src.cli.main_cli predict-batch \\
        --checkpoint best.ckpt --input-dir cifs/
    python -m src.cli.main_cli report   --checkpoint best.ckpt
    python -m src.cli.main_cli clean-cache
    python -m src.cli.main_cli doctor
"""

import argparse
import glob
import importlib
import logging
import os
import sys
from typing import List

from src.config.dataset_config import load_dataset_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

COMMANDS = (
    "dataset",
    "preprocess",
    "train",
    "evaluate",
    "explain",
    "predict",
    "predict-batch",
    "report",
    "clean-cache",
    "doctor",
)


def _run_dataset(argv: List[str]) -> None:
    """Delegates to ``dataset_cli.py``."""
    from src.cli.dataset_cli import main as dataset_main

    sys.argv = ["project dataset"] + argv
    dataset_main()


def _run_preprocess(argv: List[str]) -> None:
    """Shortcut: validates then builds graphs."""
    from src.cli.dataset_cli import (
        handle_build_graphs,
        handle_validate,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--smoke-test", type=int, default=None)
    args = parser.parse_args(argv)

    handle_validate(args)
    handle_build_graphs(args)


def _run_train(argv: List[str]) -> None:
    from src.cli.train_cli import main as train_main

    sys.argv = ["project train"] + argv
    train_main()


def _run_evaluate(argv: List[str]) -> None:
    from src.cli.evaluate_cli import main as evaluate_main

    sys.argv = ["project evaluate"] + argv
    evaluate_main()


def _run_explain(argv: List[str]) -> None:
    from src.cli.explain_cli import main as explain_main

    sys.argv = ["project explain"] + argv
    explain_main()


def _run_predict(argv: List[str]) -> None:
    """Delegates single prediction to ``predict_cli.py predict``."""
    from src.cli.predict_cli import main as predict_main

    sys.argv = ["project predict", "predict"] + argv
    predict_main()


def _run_predict_batch(argv: List[str]) -> None:
    """Delegates batch prediction to ``predict_cli.py predict-batch``."""
    from src.cli.predict_cli import main as predict_main

    sys.argv = ["project predict-batch", "predict-batch"] + argv
    predict_main()


def _run_report(argv: List[str]) -> None:
    from src.cli.report_cli import main as report_main

    sys.argv = ["project report"] + argv
    report_main()


def _run_clean_cache(argv: List[str]) -> None:
    """Removes processed graph shards so they are regenerated on next run."""
    parser = argparse.ArgumentParser(
        description="Delete cached graph shards (datasets/processed/)."
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to dataset config (used to locate the processed directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting.",
    )
    args = parser.parse_args(argv)

    config = load_dataset_config(args.config)
    processed = config.processed_data_dir

    if not os.path.isdir(processed):
        print(f"Nothing to clean: '{processed}' does not exist.")
        return

    shards = glob.glob(os.path.join(processed, "*.pt"))
    manifests = glob.glob(os.path.join(processed, "manifest*.json"))
    targets = shards + manifests

    if not targets:
        print(f"Cache directory '{processed}' is already empty.")
        return

    if args.dry_run:
        print(f"Would delete {len(targets)} file(s) from '{processed}':")
        for t in sorted(targets):
            print(f"  {t}")
        return

    for t in targets:
        os.remove(t)
    print(f"Deleted {len(targets)} cached file(s) from '{processed}'.")


def _run_doctor(argv: List[str]) -> None:
    """Runs environment and configuration health checks."""
    parser = argparse.ArgumentParser(
        description="Check project health: dependencies, configs, cache."
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to dataset config.",
    )
    args = parser.parse_args(argv)

    issues: List[str] = []
    ok: List[str] = []

    # 1. Python version
    v = sys.version_info
    if v >= (3, 12):
        ok.append(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        issues.append(f"Python >= 3.12 required, found {v.major}.{v.minor}.{v.micro}")

    # 2. Core imports
    for mod_name, label in [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("pymatgen", "pymatgen"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("captum", "Captum"),
        ("ijson", "ijson"),
    ]:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "installed")
            ok.append(f"{label} {version}")
        except ImportError:
            issues.append(f"{label} ({mod_name}) not installed")

    # 3. CUDA
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            ok.append(f"CUDA {torch.version.cuda} ({gpu})")
        else:
            ok.append("CUDA not available (CPU only)")
    except Exception:
        ok.append("CUDA check skipped")

    # 4. Config files
    for cfg in [
        "configs/dataset.yaml",
        "configs/model.yaml",
        "configs/experiment.yaml",
    ]:
        if os.path.isfile(cfg):
            ok.append(f"Config {cfg}")
        else:
            issues.append(f"Config {cfg} missing")

    # 5. Dataset
    config = load_dataset_config(args.config)
    if os.path.isfile(config.raw_data_path):
        size_mb = os.path.getsize(config.raw_data_path) / (1024 * 1024)
        ok.append(f"Dataset {config.raw_data_path} ({size_mb:.1f} MB)")
    else:
        issues.append(f"Dataset not found at {config.raw_data_path}")

    # 6. Graph cache
    if os.path.isdir(config.processed_data_dir):
        shard_count = len(glob.glob(os.path.join(config.processed_data_dir, "*.pt")))
        if shard_count > 0:
            ok.append(f"Graph cache: {shard_count} shard(s)")
        else:
            issues.append("Graph cache directory exists but contains no shards")
    else:
        issues.append(f"Graph cache directory not found: {config.processed_data_dir}")

    # Report
    print("\n" + "=" * 60)
    print("              PROJECT DOCTOR")
    print("=" * 60)
    for item in ok:
        print(f"  [OK] {item}")
    for item in issues:
        print(f"  [!!] {item}")
    print("-" * 60)
    if issues:
        print(f"  {len(issues)} issue(s) found.")
    else:
        print("  All checks passed.")
    print("=" * 60 + "\n")

    if issues:
        sys.exit(1)


_DISPATCH = {
    "dataset": _run_dataset,
    "preprocess": _run_preprocess,
    "train": _run_train,
    "evaluate": _run_evaluate,
    "explain": _run_explain,
    "predict": _run_predict,
    "predict-batch": _run_predict_batch,
    "report": _run_report,
    "clean-cache": _run_clean_cache,
    "doctor": _run_doctor,
}


def main() -> None:
    """Unified CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="project",
        description=(
            "Explainable CGCNN pipeline for formation energy prediction. "
            "Run 'project <command> --help' for command-specific options."
        ),
    )
    parser.add_argument(
        "command",
        choices=sorted(COMMANDS),
        help="Pipeline subcommand.",
    )

    args, remaining = parser.parse_known_args()
    handler = _DISPATCH[args.command]
    handler(remaining)


if __name__ == "__main__":
    main()
