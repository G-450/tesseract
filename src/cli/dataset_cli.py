import argparse
import logging
import sys

from src.config.dataset_config import load_dataset_config
from src.data.cache import load_metadata, save_metadata
from src.data.discovery import compute_fingerprint, discover_dataset
from src.data.pipeline import GraphPipeline
from src.data.validator import DatasetValidator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def handle_validate(args: argparse.Namespace) -> None:
    config = load_dataset_config(args.config)
    try:
        discovery_info = discover_dataset(config)
    except Exception as e:
        logger.error(f"Dataset discovery failed: {e}")
        sys.exit(1)

    logger.info(
        f"Discovered dataset at {discovery_info['path']} "
        f"({discovery_info['size_bytes']} bytes)"
    )

    logger.info("Computing fingerprint...")
    fingerprint = compute_fingerprint(config.raw_data_path)
    logger.info(f"Fingerprint: {fingerprint}")

    validator = DatasetValidator(config)
    stats = validator.validate(config.raw_data_path)

    save_metadata(config, stats, fingerprint)

    logger.info("Validation complete. Metadata saved.")

    print("\n=== Validation Report ===")
    print(f"Dataset: {config.raw_data_path}")
    print(f"Status: {stats['validation_status']}")
    print(f"Total records: {stats['total_records']}")
    print(f"Valid records: {stats['valid_records']}")
    print(f"Invalid records: {stats['invalid_records']}")
    print(f"Duplicate representations: {stats['duplicate_representations']}")
    print(f"Malformed structures: {stats['malformed_structures']}")
    print(f"Invalid/NaN targets: {stats['invalid_targets']}")
    print(f"Missing fields: {stats['missing_fields']}")


def handle_info(args: argparse.Namespace) -> None:
    config = load_dataset_config(args.config)

    try:
        discover_dataset(config)
    except Exception as e:
        logger.error(f"Dataset not available: {e}")
        sys.exit(1)

    fingerprint = compute_fingerprint(config.raw_data_path)
    stats = load_metadata(config, fingerprint)

    if not stats:
        logger.error(
            "No valid cached metadata found for the current dataset. "
            "Please run 'dataset validate' first."
        )
        sys.exit(1)

    print("\n=== Dataset Statistics ===")
    print(f"Dataset Path: {config.raw_data_path}")
    print(f"Validation Status: {stats.get('validation_status')}")
    print(f"Total records: {stats.get('total_records')}")
    print(f"Valid records: {stats.get('valid_records')}")
    print(f"Target Min: {stats.get('target_min')}")
    print(f"Target Max: {stats.get('target_max')}")
    print(f"Target Mean: {stats.get('target_mean')}")
    print(f"Target Std Dev: {stats.get('target_std')}")


def handle_build_graphs(args: argparse.Namespace) -> None:
    config = load_dataset_config(args.config)

    try:
        discover_dataset(config)
    except Exception as e:
        logger.error(f"Dataset not available: {e}")
        sys.exit(1)

    pipeline = GraphPipeline(config)

    if args.smoke_test is not None:
        logger.info(f"Running smoke test for {args.smoke_test} records...")

    report = pipeline.process(max_records=args.smoke_test)

    print("\n=== Graph Building Report ===")
    print(f"Records processed: {report['total_records_processed']}")
    print(f"Graphs created: {report['successful_graphs']}")
    print(f"Graphs failed: {report['failed_graphs']}")
    print(f"Shards created: {report['shards_created']}")
    print(f"Processing time: {report['processing_time_seconds']:.2f} s")
    if report["successful_graphs"] > 0:
        print(f"Average nodes per graph: {report['average_nodes']:.2f}")
        print(f"Average edges per graph: {report['average_edges']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset CLI commands")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to the dataset config YAML file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    val_parser = subparsers.add_parser(
        "validate", help="Validate the dataset and cache statistics"
    )
    val_parser.set_defaults(func=handle_validate)

    info_parser = subparsers.add_parser(
        "info", help="Print dataset statistics from cache"
    )
    info_parser.set_defaults(func=handle_info)

    build_parser = subparsers.add_parser(
        "build-graphs", help="Build crystal graphs from the dataset"
    )
    build_parser.add_argument(
        "--smoke-test",
        type=int,
        default=None,
        help="Process only the first N records",
    )
    build_parser.set_defaults(func=handle_build_graphs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
