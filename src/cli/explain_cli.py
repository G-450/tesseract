"""explain_cli.py — CLI entry point for Phase 8 Explainability.

Runs attribution methods (Integrated Gradients or GNNExplainer) on a
representative sample of test-set crystals and exports per-material JSON
and PNG artifacts (PRD §121, Phase 8 deliverables).

Usage::

    python src/cli/explain_cli.py \\
        --checkpoint experiments/checkpoints/best.ckpt \\
        --output-dir experiments/run_001/explanations \\
        --num-samples 50 \\
        --method integrated_gradients

Note:
    Outputs are model-level interpretations, not physical causal evidence
    (PRD §82, §101).
"""

import argparse
import logging
import os
import sys
from typing import List

from torch_geometric.data import Data

from src.evaluation.evaluator import Evaluator
from src.explainability.base import BaseExplainer
from src.explainability.exporter import Exporter
from src.explainability.gnn_explainer import GNNExplainerWrapper
from src.explainability.integrated_gradients import IntegratedGradientsExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate explainability reports for a trained CGCNN model. "
            "Outputs are model-level interpretations, not physical proof."
        )
    )
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
        help="Path to the dataset YAML configuration.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to the model YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where explanation artifacts will be written.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of test-set samples to explain (default: 50).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["integrated_gradients", "gnn_explainer"],
        default="integrated_gradients",
        help=(
            "Attribution method to use. "
            "'integrated_gradients' requires captum; "
            "'gnn_explainer' requires torch_geometric.explain "
            "(default: integrated_gradients)."
        ),
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
        default=0,
        help="Number of workers for the dataloader.",
    )
    return parser


def main() -> None:
    """Main CLI entry point for Phase 8 explainability."""
    parser = _build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # ── Load model and dataloader via Evaluator ───────────────────────────────
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        model_config_path=args.model_config,
        device=args.device,
        num_workers=args.num_workers,
    )
    model = evaluator.load_model()
    dataloader = evaluator.get_dataloader()

    # ── Collect requested number of single-graph samples ──────────────────────
    graphs: List[Data] = []
    for batch in dataloader:
        try:
            batch_list = batch.to_data_list()
        except AttributeError:
            # Fallback: treat the whole batch as a single item
            batch_list = [batch]

        for i, g in enumerate(batch_list):
            if not hasattr(g, "sample_id"):
                sid = getattr(g, "dataset_record_id", len(graphs) + i)
                if hasattr(sid, "item"):
                    sid = sid.item()
                g.sample_id = sid
        graphs.extend(batch_list)

        if len(graphs) >= args.num_samples:
            break

    graphs = graphs[: args.num_samples]
    logger.info(
        f"Collected {len(graphs)} samples. "
        f"Running '{args.method}' explainability..."
    )

    # ── Select and run explainer ──────────────────────────────────────────────
    explainer: BaseExplainer
    if args.method == "integrated_gradients":
        explainer = IntegratedGradientsExplainer(model)
    else:
        explainer = GNNExplainerWrapper(model)

    results = explainer.explain_batch(graphs)

    if not results:
        logger.error("No explanations were generated. Check the log for errors.")
        sys.exit(1)

    # ── Export per-material artifacts and summary CSV ─────────────────────────
    logger.info(f"Exporting {len(results)} explanations to '{args.output_dir}'...")
    Exporter.export(results, args.output_dir)

    print("\n" + "=" * 60)
    print("       CGCNN EXPLAINABILITY REPORT")
    print("=" * 60)
    print(f" Method         : {args.method}")
    print(f" Samples        : {len(results)} / {len(graphs)}")
    print(f" Output Dir     : {args.output_dir}")
    print(
        " DISCLAIMER     : Outputs are model-level interpretations,\n"
        "                  not physical causal evidence."
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
