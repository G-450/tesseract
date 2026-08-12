import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.config.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class ShardedGraphDataset(Dataset):
    """Dataset for loading sharded graph data."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.processed_dir = os.path.join(self.config.processed_data_dir, "graphs")
        self.manifest_path = os.path.join(self.processed_dir, "manifest.json")

        self.manifest: Dict[str, Any] = {}
        self.shard_files: List[str] = []
        self.shard_lengths: List[int] = []
        self.total_graphs = 0
        self.global_to_shard: List[Tuple[int, int]] = []
        
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        shard_files = sorted(
            [f for f in os.listdir(self.processed_dir) if f.startswith("shard_") and f.endswith(".pt")]
        )
        
        if not shard_files:
            raise ValueError(f"No shard files found in {self.processed_dir}")

        for shard_file in shard_files:
            shard_path = os.path.join(self.processed_dir, shard_file)
            self.shard_files.append(shard_path)

            # We load the shard just to get its length.
            # In a very large dataset, the manifest should store the length of each shard to avoid this.
            # Since the PRD indicates we are building a robust framework, we'll read it once to build the index.
            try:
                shard_data = torch.load(shard_path, weights_only=False)
                length = len(shard_data)
                self.shard_lengths.append(length)
                
                shard_idx = len(self.shard_files) - 1
                for local_idx in range(length):
                    self.global_to_shard.append((shard_idx, local_idx))
                    
            except Exception as e:
                logger.error(f"Failed to load shard {shard_path} for indexing: {e}")
                raise

        self.total_graphs = len(self.global_to_shard)
        logger.info(f"Loaded {self.total_graphs} graphs across {len(self.shard_files)} shards.")

    def __len__(self) -> int:
        return self.total_graphs

    @lru_cache(maxsize=4)
    def _load_shard(self, shard_idx: int) -> List[Data]:
        """Loads a shard into memory. Cached for fast sequential/batched access."""
        shard_path = self.shard_files[shard_idx]
        return torch.load(shard_path, weights_only=False)

    def __getitem__(self, idx: int) -> Data:
        shard_idx, local_idx = self.global_to_shard[idx]
        shard_data = self._load_shard(shard_idx)
        return shard_data[local_idx]
