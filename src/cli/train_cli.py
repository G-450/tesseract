"""
CLI script to train the CGCNN model.

Training writes every artifact into the run directory defined by
``configs/experiment.yaml`` — ``experiments/<runs_dirname>/<run_id>/`` — so
checkpoints, logs and the configuration snapshot of one run stay together and
can never overwrite another run (PRD §18, §19).
"""

import argparse
import dataclasses
import logging
import os
import shutil

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.config.dataset_config import load_dataset_config
from src.config.experiment_config import load_experiment_config
from src.config.model_config import load_cgcnn_config
from src.data.datamodule import CrystalDataModule
from src.models.lightning_module import CGCNNLightningModule
from src.utils.run_layout import RunLayout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Maximize RTX 4060 Tensor Core GPU utilization
torch.set_float32_matmul_precision("medium")


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
        "--experiment-config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment config (run identity, seed, artifact layout)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run identifier. Overrides experiment.run_id; when neither is set, "
            "one is generated as '<name>_<UTC timestamp>'"
        ),
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.ckpt) to resume training from",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader.num_workers from the dataset config",
    )
    args = parser.parse_args()

    # Load configurations
    experiment_config = load_experiment_config(args.experiment_config)
    dataset_config = load_dataset_config(args.dataset_config)
    model_config = load_cgcnn_config(args.model_config)
    if args.num_workers is not None:
        dataset_config.dataloader.num_workers = args.num_workers

    # Seeding every RNG before anything is constructed is what makes a run
    # replayable from its recorded seed (PRD §58, §17).
    seed_everything(experiment_config.seed, workers=True)

    run_id = experiment_config.resolve_run_id(args.run_id)
    layout = RunLayout.for_run(experiment_config.runs_root, run_id).create()
    layout.snapshot_configs(
        {
            "dataset": args.dataset_config,
            "model": args.model_config,
            "experiment": args.experiment_config,
        }
    )
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {layout.root}")
    logger.info(f"Seed: {experiment_config.seed}")
    logger.info(f"Model config: {dataclasses.asdict(model_config)}")

    # Instantiate DataModule and Model
    datamodule = CrystalDataModule(config=dataset_config, seed=experiment_config.seed)
    model = CGCNNLightningModule(config=model_config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=layout.checkpoints_dir,
        filename="cgcnn-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
    )

    # The learning rate is a required figure (PRD §95), so it has to be
    # logged during training rather than reconstructed afterwards.
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Loggers (TensorBoard for graphs, CSV for raw data)
    logger_tb = TensorBoardLogger(save_dir=layout.logs_dir, name="tensorboard")
    logger_csv = CSVLogger(save_dir=layout.logs_dir, name="csv")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        logger=[logger_tb, logger_csv],
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        accelerator="auto",
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule, ckpt_path=args.resume)
    logger.info("Training complete.")

    # Copy the best checkpoint to a stable path for evaluation
    best_path = checkpoint_callback.best_model_path
    if best_path:
        best_copy = os.path.join(layout.checkpoints_dir, "best.ckpt")
        shutil.copyfile(best_path, best_copy)
        logger.info(f"Best checkpoint ({best_path}) copied to {best_copy}")

    print("\n" + "=" * 68)
    print("                        TRAINING COMPLETE")
    print("=" * 68)
    print(f" Run ID        : {run_id}")
    print(f" Run directory : {layout.root}")
    print(f" Checkpoints   : {layout.checkpoints_dir}")
    print(f" Logs          : {layout.logs_dir}")
    print("-" * 68)
    print(" Next step: generate the experiment report")
    print(
        f"   python src/cli/report_cli.py --run-id {run_id} "
        f"--checkpoint {os.path.join(layout.checkpoints_dir, 'best.ckpt')}"
    )
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
