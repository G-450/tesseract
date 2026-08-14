import json
import logging
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data  # type: ignore

from src.config.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class ShardedGraphDataset(IterableDataset):
    """Shard-sequential IterableDataset for crystal graph data.

    Graphs are yielded shard-by-shard in shard order, with optional
    intra-shard shuffling (for train splits). This avoids the cache-thrashing
    caused by random global index permutation (random_split + shuffle=True),
    where a single batch of 32 items can touch 32 different shards.

    Splits are **contiguous shard ranges** (e.g. shards 0–79 for train),
    not random index permutations, so a single shard read serves all graphs
    in that shard with zero evictions.

    Multi-worker safe: :func:`torch.utils.data.get_worker_info` is used
    inside ``__iter__`` to partition the shard list across DataLoader workers,
    ensuring each shard is loaded exactly once across all workers.

    Initialisation is O(1) in dataset size: shard lengths are read from
    ``manifest.json`` (written by :class:`~src.data.pipeline.GraphPipeline`)
    rather than loaded from the ``.pt`` files.
    """

    def __init__(
        self,
        config: DatasetConfig,
        shard_files: List[str],
        shard_lengths: List[int],
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.config = config
        self.shard_files = shard_files
        self.shard_lengths = shard_lengths
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch counter for per-epoch deterministic shuffling.

        Call this at the start of every training epoch so that each epoch
        sees a different intra-shard permutation while remaining fully
        reproducible. In a Lightning workflow this is best wired into
        ``CrystalDataModule.on_train_epoch_start``.
        """
        self._epoch = epoch

    def __len__(self) -> int:
        """Total number of graphs across all shards in this split."""
        return sum(self.shard_lengths)

    def __iter__(self) -> Iterator[Data]:
        worker_info = get_worker_info()
        if worker_info is not None:
            # Interleave shards across workers so every shard is owned by
            # exactly one worker — no duplicate disk reads.
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shard_files: List[str] = self.shard_files[worker_id::num_workers]
        else:
            shard_files = self.shard_files

        for abs_shard_idx, shard_path in enumerate(shard_files):
            graphs: List[Data] = list(torch.load(shard_path, weights_only=False))
            if self.shuffle:
                # Seed = epoch * #shards + shard position, giving a unique
                # permutation per epoch per shard without global state.
                rng = random.Random(self._epoch * len(self.shard_files) + abs_shard_idx)
                rng.shuffle(graphs)
            yield from graphs

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def split(
        cls,
        config: DatasetConfig,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple["ShardedGraphDataset", "ShardedGraphDataset", "ShardedGraphDataset"]:
        """Split shards into train / val / test contiguous ranges.

        Splits are computed at the **shard** level so that the train
        DataLoader never needs to cross a shard boundary to form a batch.

        Args:
            config:       Dataset configuration.
            train_ratio:  Fraction of shards allocated to training.
            val_ratio:    Fraction of shards allocated to validation.
            test_ratio:   Fraction of shards allocated to testing.
            seed:         RNG seed for intra-shard shuffling in train split.

        Returns:
            A triple ``(train_dataset, val_dataset, test_dataset)``.
        """
        _, shard_files, shard_lengths = cls._load_index(config)
        n = len(shard_files)

        total = train_ratio + val_ratio + test_ratio
        train_frac = train_ratio / total
        val_frac = val_ratio / total

        train_end = max(1, int(n * train_frac)) if n >= 1 else 0
        val_end = min(train_end + int(n * val_frac), n)

        train_files = shard_files[:train_end]
        train_lengths = shard_lengths[:train_end]
        val_files = shard_files[train_end:val_end]
        val_lengths = shard_lengths[train_end:val_end]
        test_files = shard_files[val_end:]
        test_lengths = shard_lengths[val_end:]

        logger.info(
            "Shard split: %d train (%d graphs), %d val (%d graphs), "
            "%d test (%d graphs)",
            len(train_files),
            sum(train_lengths),
            len(val_files),
            sum(val_lengths),
            len(test_files),
            sum(test_lengths),
        )

        return (
            cls(config, train_files, train_lengths, shuffle=True, seed=seed),
            cls(config, val_files, val_lengths, shuffle=False, seed=seed),
            cls(config, test_files, test_lengths, shuffle=False, seed=seed),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_index(
        config: DatasetConfig,
    ) -> Tuple[str, List[str], List[int]]:
        """Load the ordered shard file list and per-shard lengths.

        Reads lengths from ``manifest["shard_lengths"]`` (O(1) I/O).
        Falls back to loading each shard once if the manifest predates
        the ``shard_lengths`` field (legacy), emitting a warning.

        Returns:
            ``(processed_dir, shard_files, shard_lengths)``
        """
        processed_dir = os.path.join(config.processed_data_dir, "graphs")
        manifest_path = os.path.join(processed_dir, "manifest.json")

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest: Dict[str, Any] = json.load(f)

        shard_filenames = sorted(
            fname
            for fname in os.listdir(processed_dir)
            if fname.startswith("shard_") and fname.endswith(".pt")
        )
        if not shard_filenames:
            raise ValueError(f"No shard files found in {processed_dir}")

        manifest_lengths: Optional[Dict[str, int]] = manifest.get("shard_lengths")

        shard_files: List[str] = []
        shard_lengths: List[int] = []

        for fname in shard_filenames:
            shard_path = os.path.join(processed_dir, fname)
            shard_files.append(shard_path)

            if manifest_lengths is not None:
                length = manifest_lengths.get(fname)
                if length is None:
                    raise KeyError(
                        f"Shard '{fname}' missing from manifest shard_lengths. "
                        "Re-run the pipeline to regenerate the manifest."
                    )
                shard_lengths.append(length)
            else:
                # Legacy manifest: fall back to loading the shard once (slow).
                logger.warning(
                    "Manifest at '%s' does not contain 'shard_lengths'. "
                    "Loading '%s' to determine length (legacy fallback). "
                    "Re-run the pipeline to produce a fast manifest.",
                    manifest_path,
                    fname,
                )
                shard_data = torch.load(shard_path, weights_only=False)
                shard_lengths.append(len(shard_data))

        logger.info(
            "Indexed %d shards, %d total graphs.",
            len(shard_files),
            sum(shard_lengths),
        )
        return processed_dir, shard_files, shard_lengths
