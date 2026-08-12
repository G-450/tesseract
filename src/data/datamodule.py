import logging
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader  # type: ignore

from src.config.dataset_config import DatasetConfig
from src.data.dataset import ShardedGraphDataset

logger = logging.getLogger(__name__)


class CrystalDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the crystal graph dataset."""

    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.config = config
        self.dataset: Optional[ShardedGraphDataset] = None
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            logger.info("Initializing ShardedGraphDataset...")
            self.dataset = ShardedGraphDataset(self.config)

            total_len = len(self.dataset)
            train_ratio = self.config.dataloader.train_split
            val_ratio = self.config.dataloader.val_split
            test_ratio = self.config.dataloader.test_split

            # Normalize ratios just in case
            total_ratio = train_ratio + val_ratio + test_ratio
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio

            train_len = int(total_len * train_ratio)
            val_len = int(total_len * val_ratio)
            test_len = total_len - train_len - val_len

            # We use a fixed generator for deterministic splits across runs
            import torch

            generator = torch.Generator().manual_seed(42)

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, [train_len, val_len, test_len], generator=generator
            )

            logger.info(
                f"Dataset split: {train_len} train, {val_len} val, {test_len} test."
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True,
        )
