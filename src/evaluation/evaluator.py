"""Model evaluator module for formation energy prediction."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch_geometric.loader import DataLoader  # type: ignore

from src.config.dataset_config import DatasetConfig, load_dataset_config
from src.config.model_config import CGCNNConfig, load_cgcnn_config
from src.data.datamodule import CrystalDataModule
from src.evaluation.metrics import compute_all_metrics
from src.models.lightning_module import CGCNNLightningModule
from src.utils.checkpoints import checkpoint_safe_globals

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Dataclass holding complete evaluation results and sample-level outputs.

    Attributes:
        predictions: 1D Tensor of model predictions.
        targets: 1D Tensor of ground truth values.
        metrics: Dictionary of computed scalar metrics (mae, rmse, r2, bias).
        sample_ids: List of sample identifiers or dataset record IDs.
        residuals: 1D Tensor of residuals (predictions - targets).
        absolute_errors: 1D Tensor of absolute errors (|predictions - targets|).
    """

    predictions: torch.Tensor
    targets: torch.Tensor
    metrics: Dict[str, float]
    sample_ids: List[Any] = field(default_factory=list)
    residuals: torch.Tensor = field(init=False)
    absolute_errors: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.predictions = self.predictions.detach().cpu().view(-1).float()
        self.targets = self.targets.detach().cpu().view(-1).float()
        self.residuals = (self.predictions - self.targets).float()
        self.absolute_errors = torch.abs(self.residuals).float()

        if not self.sample_ids:
            self.sample_ids = list(range(len(self.predictions)))


class Evaluator:
    """Evaluates a trained CGCNN model checkpoint on the dataset test split."""

    def __init__(
        self,
        checkpoint_path: str,
        dataset_config_path: str = "configs/dataset.yaml",
        model_config_path: str = "configs/model.yaml",
        device: Optional[str] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        """Initializes the Evaluator with checkpoint and configuration paths.

        Args:
            checkpoint_path: Path to the PyTorch Lightning .ckpt checkpoint file.
            dataset_config_path: Path to dataset YAML configuration.
            model_config_path: Path to model YAML configuration.
            device: Optional compute device ('cpu', 'cuda', etc.). If None,
                defaults to 'cuda' if available else 'cpu'.
            num_workers: Optional override for the dataset config's
                dataloader.num_workers.
        """
        self.checkpoint_path = checkpoint_path
        self.dataset_config_path = dataset_config_path
        self.model_config_path = model_config_path

        self.dataset_config: DatasetConfig = load_dataset_config(dataset_config_path)
        self.model_config: CGCNNConfig = load_cgcnn_config(model_config_path)
        if num_workers is not None:
            self.dataset_config.dataloader.num_workers = num_workers

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model: Optional[CGCNNLightningModule] = None

    def load_model(self) -> CGCNNLightningModule:
        """Loads and returns the model from checkpoint in evaluation mode.

        Returns:
            CGCNNLightningModule: Model ready for inference.
        """
        if self._model is None:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}...")
            with checkpoint_safe_globals():
                self._model = CGCNNLightningModule.load_from_checkpoint(
                    self.checkpoint_path,
                    config=self.model_config,
                )
            self._model.to(self.device)
            self._model.eval()
            self._model.freeze()
        return self._model

    def get_dataloader(self) -> DataLoader:
        """Constructs and returns the test split DataLoader.

        Returns:
            DataLoader: PyG DataLoader for the test dataset.
        """
        datamodule = CrystalDataModule(config=self.dataset_config)
        datamodule.setup(stage="test")
        return datamodule.test_dataloader()

    def evaluate_dataloader(
        self,
        dataloader: DataLoader,
        model: Optional[CGCNNLightningModule] = None,
    ) -> EvaluationResult:
        """Evaluates a model over a provided DataLoader.

        Args:
            dataloader: PyTorch Geometric DataLoader yielding graph batches.
            model: Optional model instance. If None, uses loaded checkpoint model.

        Returns:
            EvaluationResult: Evaluation results containing predictions and metrics.
        """
        eval_model = model if model is not None else self.load_model()
        eval_model.eval()

        predictions_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        sample_ids_list: List[Any] = []

        logger.info("Running evaluation forward pass...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                out = eval_model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                pred = out.view(-1).detach().cpu()
                target = batch.y.float().view(-1).detach().cpu()

                predictions_list.append(pred)
                targets_list.append(target)

                # Extract sample / record IDs if present on batch
                if hasattr(batch, "dataset_record_id"):
                    record_ids = batch.dataset_record_id
                    if isinstance(record_ids, torch.Tensor):
                        sample_ids_list.extend(record_ids.cpu().tolist())
                    elif isinstance(record_ids, list):
                        sample_ids_list.extend(record_ids)
                    else:
                        sample_ids_list.extend([record_ids] * len(pred))
                else:
                    sample_ids_list.extend(
                        range(
                            len(sample_ids_list),
                            len(sample_ids_list) + len(pred),
                        )
                    )

        if not predictions_list:
            raise ValueError("Test DataLoader produced no batches.")

        all_preds = torch.cat(predictions_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        metrics = compute_all_metrics(all_preds, all_targets)

        logger.info(
            f"Evaluation complete. MAE: {metrics['mae']:.4f} eV/atom, "
            f"RMSE: {metrics['rmse']:.4f} eV/atom, R²: {metrics['r2']:.4f}, "
            f"Bias: {metrics['bias']:.4f} eV/atom"
        )

        return EvaluationResult(
            predictions=all_preds,
            targets=all_targets,
            metrics=metrics,
            sample_ids=sample_ids_list,
        )

    def run(self) -> EvaluationResult:
        """Runs evaluation on the test dataset split.

        Returns:
            EvaluationResult: Evaluation results with all predictions and metrics.
        """
        dataloader = self.get_dataloader()
        return self.evaluate_dataloader(dataloader)
