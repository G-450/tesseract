import logging
import os

import torch
from torch_geometric.data import InMemoryDataset

from src.config.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class ShardedGraphDataset(InMemoryDataset):
    """Dataset for loading sharded graph data into contiguous memory."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.graph_dir = os.path.join(self.config.processed_data_dir, "graphs")
        self.manifest_path = os.path.join(self.graph_dir, "manifest.json")
        self.compiled_path = os.path.join(self.graph_dir, "compiled_dataset.pt")

        super().__init__(root=self.config.processed_data_dir)

        if not os.path.exists(self.compiled_path):
            self._compile_shards()

        # Load the contiguous compiled dataset
        logger.info("Loading compiled dataset into memory...")
        self.data, self.slices = torch.load(self.compiled_path, weights_only=False)
        logger.info(f"Loaded {len(self)} graphs efficiently into RAM.")

    def _compile_shards(self) -> None:
        """Collates all shards into a single contiguous InMemoryDataset."""
        logger.info("Collating fragmented shards into a single memory-mapped tensor...")
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        shard_files = sorted(
            [
                f
                for f in os.listdir(self.graph_dir)
                if f.startswith("shard_") and f.endswith(".pt")
            ]
        )

        all_data_list = []
        for shard_file in shard_files:
            shard_path = os.path.join(self.graph_dir, shard_file)
            try:
                shard_data = torch.load(shard_path, weights_only=False)
                all_data_list.extend(shard_data)
            except Exception as e:
                logger.error(f"Failed to load shard {shard_path}: {e}")

        # Collate into single Data object with slices
        data, slices = self.collate(all_data_list)
        torch.save((data, slices), self.compiled_path)
        logger.info(
            f"Successfully compiled {len(all_data_list)} graphs to {self.compiled_path}"
        )
