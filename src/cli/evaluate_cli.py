"""CLI script to evaluate a trained CGCNN model checkpoint."""

import argparse
import logging
import os
import sys

from src.evaluation.evaluator import Evaluator
from src.evaluation.reporter import Reporter
from src.visualization.plots import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CGCNN model on the test dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/evaluation_001",
        help="Directory to save evaluation reports and publication plots",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="experiments/cgcnn_csv_logs",
        help="Path to training CSV logs directory (for plotting loss curves)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit compute device ('cpu', 'cuda', etc.)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader.num_workers from the dataset config",
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found at: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"Initializing evaluation for checkpoint: {args.checkpoint}")
    logger.info(f"Dataset config: {args.dataset_config}")
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Output directory: {args.output_dir}")

    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        model_config_path=args.model_config,
        device=args.device,
        num_workers=args.num_workers,
    )

    logger.info("Executing evaluation pipeline...")
    result = evaluator.run()

    logger.info(f"Exporting evaluation reports to {args.output_dir}...")
    reporter = Reporter()
    exported_reports = reporter.export(result, args.output_dir)

    logger.info("Generating publication-quality plots...")
    exported_plots = generate_all_plots(
        result=result,
        output_dir=args.output_dir,
        log_dir=args.log_dir if os.path.exists(args.log_dir) else None,
    )

    # Print summary table to console
    print("\n" + "=" * 60)
    print("           CGCNN MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f" Total Test Samples : {len(result.predictions)}")
    print(f" MAE (eV/atom)      : {result.metrics.get('mae', 0.0):.4f}")
    print(f" RMSE (eV/atom)     : {result.metrics.get('rmse', 0.0):.4f}")
    print(f" R² Score           : {result.metrics.get('r2', 0.0):.4f}")
    print(f" Mean Bias (eV/atom): {result.metrics.get('bias', 0.0):+.4f}")
    print("-" * 60)
    print(f" Published Baseline : {reporter.baseline_mae:.4f} eV/atom")
    mae_diff = result.metrics.get("mae", 0.0) - reporter.baseline_mae
    print(f" Difference vs Base : {mae_diff:+.4f} eV/atom")
    print("-" * 60)
    print(" Exported Artifacts:")
    for name, path in {**exported_reports, **exported_plots}.items():
        print(f"   - {name}: {path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
