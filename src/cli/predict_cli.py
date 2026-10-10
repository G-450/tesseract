"""predict_cli.py -- CLI entry point for Phase 9 Inference and Prediction.

Predicts formation energy (eV/atom) for crystal structures that are not part
of the training dataset, using a trained CGCNN checkpoint (PRD 89-92, 121).

Usage::

    # Single structure
    python src/cli/predict_cli.py predict \\
        --checkpoint experiments/checkpoints/best.ckpt \\
        --input structures/NaCl.cif \\
        --output outputs/NaCl.json

    # Directory of structures
    python src/cli/predict_cli.py predict-batch \\
        --checkpoint experiments/checkpoints/best.ckpt \\
        --input-dir structures/ \\
        --output-dir outputs/predictions

Note:
    Inference reuses the training graph preprocessing exactly and never
    retrains or modifies the model (PRD 89, 90).
"""

import argparse
import logging
import sys

from src.inference.exceptions import InferenceError
from src.inference.exporter import PredictionExporter
from src.inference.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the arguments shared by every prediction subcommand."""
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
        help=(
            "Path to the dataset YAML configuration. Its 'graph' section must "
            "match the configuration used for training."
        ),
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to the model YAML configuration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device override ('cpu', 'cuda', etc.).",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Constructs and returns the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict formation energy (eV/atom) for unseen crystal structures "
            "with a trained CGCNN model."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "predict",
        help="Predict the formation energy of a single structure (CIF).",
    )
    _add_common_arguments(single)
    single.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CIF file.",
    )
    single.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional export path. The extension selects the format "
            "(.csv or .json). When omitted, the prediction is printed only."
        ),
    )

    batch = subparsers.add_parser(
        "predict-batch",
        help="Predict formation energies for a directory of structures.",
    )
    _add_common_arguments(batch)
    batch.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing CIF files to predict.",
    )
    batch.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where predictions.csv and predictions.json are written.",
    )
    batch.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Structures per forward pass. Defaults to the dataset config's "
            "dataloader.batch_size."
        ),
    )
    return parser


def _run_single(args: argparse.Namespace, predictor: Predictor) -> None:
    """Executes the 'predict' subcommand."""
    result = predictor.predict_cif(args.input)

    exported = None
    if args.output:
        exported = PredictionExporter.export_single(result, args.output)

    print("\n" + "=" * 60)
    print("       CGCNN FORMATION ENERGY PREDICTION")
    print("=" * 60)
    print(f" Structure      : {result.sample_id}")
    print(f" Formula        : {result.formula}")
    print(f" Atoms / Bonds  : {result.num_atoms} / {result.num_edges}")
    print(f" Prediction     : {result.formation_energy:.4f} eV/atom")
    print(f" Checkpoint     : {predictor.checkpoint_path}")
    print(f" Device         : {predictor.device}")
    if exported:
        print(f" Exported       : {exported}")
    print("=" * 60 + "\n")


def _run_batch(args: argparse.Namespace, predictor: Predictor) -> None:
    """Executes the 'predict-batch' subcommand."""
    result = predictor.predict_batch(args.input_dir, batch_size=args.batch_size)

    if result.num_attempted == 0:
        logger.error(f"No CIF files found in '{args.input_dir}'.")
        sys.exit(1)

    artifacts = PredictionExporter.export(result, args.output_dir)
    summary = result.summary()

    print("\n" + "=" * 60)
    print("       CGCNN BATCH PREDICTION REPORT")
    print("=" * 60)
    print(f" Input Dir      : {args.input_dir}")
    print(f" Attempted      : {summary['num_attempted']}")
    print(f" Succeeded      : {summary['num_succeeded']}")
    print(f" Skipped        : {summary['num_failed']}")
    if result.predictions:
        print(
            " Mean Energy    : "
            f"{summary['mean_formation_energy_ev_per_atom']:.4f} eV/atom"
        )
        print(
            " Range          : "
            f"{summary['min_formation_energy_ev_per_atom']:.4f} to "
            f"{summary['max_formation_energy_ev_per_atom']:.4f} eV/atom"
        )
    print(f" Checkpoint     : {predictor.checkpoint_path}")
    print(f" Device         : {predictor.device}")
    for path in artifacts:
        print(f" Exported       : {path}")
    print("=" * 60 + "\n")

    if result.failures:
        print("Skipped structures:")
        for sample_id, reason in result.failures:
            print(f"  - {sample_id}: {reason}")
        print()


def main() -> None:
    """Main CLI entry point for Phase 9 inference."""
    parser = _build_parser()
    args = parser.parse_args()

    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        model_config_path=args.model_config,
        device=args.device,
    )

    try:
        if args.command == "predict":
            _run_single(args, predictor)
        else:
            _run_batch(args, predictor)
    except InferenceError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
