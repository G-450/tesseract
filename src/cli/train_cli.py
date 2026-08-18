"""
CLI script to train the CGCNN model.
"""

import argparse
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.config.dataset_config import load_dataset_config
from src.config.model_config import load_cgcnn_config
from src.data.datamodule import CrystalDataModule
from src.models.lightning_module import CGCNNLightningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CGCNN model")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset config",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a fast development run (1 epoch, 1 batch)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum number of epochs to train",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Limit the number of training batches (e.g., 0.1 for 10%)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Limit the number of validation batches",
    )
    args = parser.parse_args()

    # Load configurations
    dataset_config = load_dataset_config(args.dataset_config)
    model_config = load_cgcnn_config(args.model_config)

    # Instantiate DataModule and Model
    datamodule = CrystalDataModule(config=dataset_config)
    model = CGCNNLightningModule(config=model_config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join("experiments", "checkpoints"),
        filename="cgcnn-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
    )

    # Logger
    logger_tb = TensorBoardLogger("experiments", name="cgcnn_logs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        logger=logger_tb,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="auto",
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
