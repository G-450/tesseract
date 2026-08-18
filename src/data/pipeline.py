import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import ijson  # type: ignore
import torch
from pymatgen.core.structure import Structure

from src.config.dataset_config import DatasetConfig
from src.graphs.builder import CrystalGraphBuilder

logger = logging.getLogger(__name__)


class GraphPipeline:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.builder = CrystalGraphBuilder(config.graph)

    def _compute_fingerprint(self, filepath: str) -> str:
        """Computes SHA-256 fingerprint of the file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192 * 1024):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _save_shard(
        self, shard_data: List[Any], shard_idx: int, output_dir: str
    ) -> str:
        """Saves a shard to disk."""
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
        # Save atomically to prevent partial writes
        temp_path = shard_path + ".tmp"
        torch.save(shard_data, temp_path)
        os.replace(temp_path, shard_path)
        return shard_path

    def process(self, max_records: Optional[int] = None) -> Dict[str, Any]:
        """
        Processes the raw dataset into graph shards.

        Args:
            max_records (int, optional): If set, only process this many
                                         records (for smoke tests).

        Returns:
            Dict: Processing report.
        """
        filepath = self.config.raw_data_path
        output_dir = os.path.join(self.config.processed_data_dir, "graphs")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Computing fingerprint for {filepath}")
        fingerprint = (
            self.compute_fingerprint(filepath)
            if max_records is None
            else "smoke-test-fingerprint"
        )

        columns: List[str] = []
        with open(filepath, "rb") as f:
            parser = ijson.parse(f, use_float=True)
            for prefix, event, value in parser:
                if prefix == "columns.item":
                    columns.append(value)
                elif prefix == "columns" and event == "end_array":
                    break

        try:
            structure_idx = columns.index("structure")
            target_idx = columns.index(self.config.target_column)
        except ValueError:
            raise ValueError(
                f"Required columns 'structure' or "
                f"'{self.config.target_column}' missing."
            )

        report: Dict[str, Any] = {
            "fingerprint": fingerprint,
            "total_records_processed": 0,
            "successful_graphs": 0,
            "failed_graphs": 0,
            "failure_details": [],
            "shards_created": 0,
            "start_time": time.time(),
            "graph_config": self.config.graph.__dict__,
            "total_nodes": 0,
            "total_edges": 0,
        }

        current_shard: List[Any] = []
        shard_idx = 0

        logger.info("Starting graph processing...")

        with open(filepath, "rb") as f:
            items = ijson.items(f, "data.item", use_float=True)
            for record_id, record in enumerate(items):
                if (
                    max_records is not None
                    and report["total_records_processed"] >= max_records
                ):
                    break

                report["total_records_processed"] += 1

                try:
                    if len(record) <= max(structure_idx, target_idx):
                        raise ValueError("Record is missing required columns.")

                    struct_data = record[structure_idx]
                    target_val = record[target_idx]

                    if target_val is None:
                        raise ValueError("Target value is missing.")

                    structure = Structure.from_dict(struct_data)

                    graph = self.builder.build(
                        structure=structure,
                        record_id=record_id,
                        fingerprint=fingerprint,
                        target=float(target_val),
                    )

                    current_shard.append(graph)
                    report["successful_graphs"] += 1
                    report["total_nodes"] += graph.num_nodes
                    report["total_edges"] += graph.num_edges

                    if len(current_shard) >= self.config.graph.shard_size:
                        self._save_shard(current_shard, shard_idx, output_dir)
                        report["shards_created"] += 1
                        shard_idx += 1
                        current_shard = []

                except Exception as e:
                    report["failed_graphs"] += 1
                    report["failure_details"].append(
                        {"record_id": record_id, "error": str(e)}
                    )
                    logger.warning(f"Failed to process record {record_id}: {e}")

        # Save any remaining graphs in the last shard
        if current_shard:
            self._save_shard(current_shard, shard_idx, output_dir)
            report["shards_created"] += 1

        report["end_time"] = time.time()
        report["processing_time_seconds"] = report["end_time"] - report["start_time"]

        if report["successful_graphs"] > 0:
            report["average_nodes"] = (
                report["total_nodes"] / report["successful_graphs"]
            )
            report["average_edges"] = (
                report["total_edges"] / report["successful_graphs"]
            )

        # Save manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            f"Graph construction complete. {report['successful_graphs']} successful, "
            f"{report['failed_graphs']} failed."
        )

        return report

    def compute_fingerprint(self, filepath: str) -> str:
        return self._compute_fingerprint(filepath)
