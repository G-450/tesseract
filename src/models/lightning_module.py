from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)

from src.config.model_config import CGCNNConfig
from src.models.cgcnn import CGCNN


class CGCNNLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for training the CGCNN model.
    """

    def __init__(self, config: CGCNNConfig):
        super(CGCNNLightningModule, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = CGCNN(config.model)

        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
                "MAE": MeanAbsoluteError(),
                "R2": R2Score(),
                "MAPE": MeanAbsolutePercentageError(),
                "Pearson": PearsonCorrCoef(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> Any:
        return self.model(x, edge_index, edge_attr, batch)

    def _shared_step(self, batch: Any, batch_idx: int, step_name: str) -> Any:
        """
        Shared forward pass and loss calculation.
        """
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        out = out.view(-1)
        target = batch.y.float()

        # Calculate loss (MSE for regression)
        loss = F.mse_loss(out, target)

        if step_name == "train":
            metrics = self.train_metrics(out, target)
        elif step_name == "val":
            metrics = self.val_metrics(out, target)
        else:
            metrics = self.test_metrics(out, target)

        self.log_dict(
            metrics,
            batch_size=batch.num_graphs,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        # Also log the primary loss directly for the progress bar at each step
        self.log(
            f"{step_name}_loss",
            loss,
            batch_size=batch.num_graphs,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> Any:
        optimizer = Adam(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.optimizer.lr_factor,
            patience=self.config.optimizer.lr_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
