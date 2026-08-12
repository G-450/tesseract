import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.cgcnn import CGCNN
from src.config.model_config import CGCNNConfig

class CGCNNLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for training the CGCNN model.
    """
    def __init__(self, config: CGCNNConfig):
        super(CGCNNLightningModule, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = CGCNN(config.model)
        
    def forward(self, x, edge_index, edge_attr, batch):
        return self.model(x, edge_index, edge_attr, batch)
        
    def _shared_step(self, batch, batch_idx, step_name):
        """
        Shared forward pass and loss calculation.
        """
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # Squeeze output to match target shape
        out = out.view(-1)
        
        # Calculate loss (MSE for regression)
        loss = F.mse_loss(out, batch.y)
        
        # Calculate MAE for metric logging
        mae = F.l1_loss(out, batch.y)
        
        self.log(f"{step_name}_loss", loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True)
        self.log(f"{step_name}_mae", mae, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), 
            lr=self.config.optimizer.lr, 
            weight_decay=self.config.optimizer.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.optimizer.lr_factor,
            patience=self.config.optimizer.lr_patience
        )

        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
