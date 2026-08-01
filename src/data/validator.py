import logging
import math
from typing import Any, Dict, List

import ijson  # type: ignore

from src.config.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class DatasetValidator:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats: Dict[str, Any] = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "missing_fields": 0,
            "invalid_targets": 0,
            "malformed_structures": 0,
            "duplicate_representations": 0,
            "target_min": float("inf"),
            "target_max": float("-inf"),
            "target_mean": 0.0,
            "target_M2": 0.0,
        }
        self.seen_structure_hashes: set[str] = set()

    def _is_valid_float(self, val: Any) -> bool:
        if val is None:
            return False
        try:
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                return False
            return True
        except (ValueError, TypeError):
            return False

    def _validate_structure(self, struct_data: Any) -> bool:
        if not isinstance(struct_data, dict):
            return False
        if struct_data.get("@class") != "Structure":
            return False
        if "lattice" not in struct_data or "sites" not in struct_data:
            return False
        return True

    def validate(self, filepath: str) -> Dict[str, Any]:
        """
        Incrementally validates the dataset without loading the entire JSON into memory.
        Expects a Pandas 'split' orient JSON with 'columns' and 'data'.

        Args:
            filepath: Path to the dataset JSON.

        Returns:
            Dict containing validation statistics and status.
        """
        self.reset_stats()

        columns: List[str] = []
        structure_idx = -1
        target_idx = -1

        logger.info(f"Starting validation for {filepath}")

        # Pass 1: find columns
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

        # Pass 2: iterate over data items
        with open(filepath, "rb") as f:
            items = ijson.items(f, "data.item", use_float=True)
            for record in items:
                self.stats["total_records"] += 1

                if len(record) <= max(structure_idx, target_idx):
                    self.stats["missing_fields"] += 1
                    self.stats["invalid_records"] += 1
                    continue

                struct_data = record[structure_idx]
                target_data = record[target_idx]

                is_valid = True

                if self.config.require_structure and not self._validate_structure(
                    struct_data
                ):
                    self.stats["malformed_structures"] += 1
                    is_valid = False

                if self.config.require_target and not self._is_valid_float(target_data):
                    self.stats["invalid_targets"] += 1
                    is_valid = False

                if is_valid:
                    import hashlib
                    import json

                    struct_json = json.dumps(
                        struct_data, sort_keys=True, separators=(",", ":")
                    )
                    struct_hash = hashlib.sha256(
                        struct_json.encode("utf-8")
                    ).hexdigest()
                    if struct_hash in self.seen_structure_hashes:
                        self.stats["duplicate_representations"] += 1
                    else:
                        self.seen_structure_hashes.add(struct_hash)

                    self.stats["valid_records"] += 1
                    val = float(target_data)
                    self.stats["target_min"] = min(self.stats["target_min"], val)
                    self.stats["target_max"] = max(self.stats["target_max"], val)

                    # Welford's online algorithm
                    delta = val - self.stats["target_mean"]
                    self.stats["target_mean"] += delta / self.stats["valid_records"]
                    delta2 = val - self.stats["target_mean"]
                    self.stats["target_M2"] += delta * delta2
                else:
                    self.stats["invalid_records"] += 1

        # Compute final statistics
        n = self.stats["valid_records"]
        if n > 0:
            variance = self.stats["target_M2"] / n
            self.stats["target_std"] = math.sqrt(variance)
        else:
            self.stats["target_min"] = None
            self.stats["target_max"] = None
            self.stats["target_mean"] = None
            self.stats["target_std"] = None

        self.stats["validation_status"] = (
            "PASS"
            if self.stats["valid_records"] > 0 and self.stats["invalid_records"] == 0
            else "FAIL"
        )

        logger.info(
            f"Validation completed: {self.stats['valid_records']} valid, "
            f"{self.stats['invalid_records']} invalid."
        )
        return self.stats
