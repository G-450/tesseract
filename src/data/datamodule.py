import logging
from typing import Optional

import pytorch_lightning as pl  # type: ignore
from pytorch_lightning.callbacks import Callback  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

from src.config.dataset_config import DatasetConfig
from src.data.dataset import ShardedGraphDataset

logger = logging.getLogger(__name__)


class AdvanceEpochCallback(Callback):
    """Callback to advance the epoch for IterableDatasets that need a seed.

    Wired here to ensure the IterableDataset shuffling gets a new seed every
    epoch, preventing the model from overfitting to a fixed shard permutation.
    """

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            if getattr(trainer.datamodule, "train_dataset", None) is not None:
                trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch)


class CrystalDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the crystal graph dataset.

    Splits are performed at the **shard** level (contiguous ranges), not via
    global random index permutation, so the DataLoader never needs to cross a
    shard boundary to serve a single batch.  Intra-shard shuffling is handled
    inside :class:`~src.data.dataset.ShardedGraphDataset` and advances its
    seed each epoch via :class:`AdvanceEpochCallback`.
    """

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: Optional[ShardedGraphDataset] = None
        self.val_dataset: Optional[ShardedGraphDataset] = None
        self.test_dataset: Optional[ShardedGraphDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is None:
            logger.info("Splitting shards into train / val / test …")
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = ShardedGraphDataset.split(
                self.config,
                train_ratio=self.config.dataloader.train_split,
                val_ratio=self.config.dataloader.val_split,
                test_ratio=self.config.dataloader.test_split,
                seed=42,
            )

    def train_dataloader(self) -> DataLoader:
        # shuffle=False: IterableDataset does not support the DataLoader
        # shuffle flag; intra-shard shuffling is handled inside __iter__.
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False,
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
