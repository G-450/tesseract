"""Formation energy prediction for unseen crystal structures (PRD 89-92).

The predictor operates independently from the training subsystem: it imports
only the model definition and the graph builder, never the trainer, the
datamodule or the processed dataset. Graphs are constructed with the exact
same ``CrystalGraphBuilder`` and ``GraphConfig`` used during training, because
any divergence between training and inference preprocessing is a critical
defect (PRD 90).

Inference never modifies or retrains the model: the checkpoint is loaded in
``eval`` mode and frozen.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from pymatgen.core.structure import Structure
from torch_geometric.data import Batch, Data  # type: ignore

from src.config.dataset_config import DatasetConfig, GraphConfig, load_dataset_config
from src.config.model_config import CGCNNConfig, load_cgcnn_config
from src.graphs.builder import CrystalGraphBuilder
from src.inference.exceptions import (
    ModelLoadError,
    PredictionError,
    StructureLoadError,
)
from src.models.lightning_module import CGCNNLightningModule
from src.utils.checkpoints import checkpoint_safe_globals

logger = logging.getLogger(__name__)

# Placeholder graph metadata for structures that are not dataset records.
# The target is unknown at inference time; the builder requires one, so a
# sentinel is supplied and never read by the forward pass.
_INFERENCE_RECORD_ID = -1
_INFERENCE_FINGERPRINT = "inference"
_INFERENCE_TARGET = 0.0

# Crystal structure file extensions recognised for batch discovery (PRD 92).
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".cif",)


@dataclass
class PredictionResult:
    """Prediction for a single crystal structure.

    Attributes:
        sample_id: Identifier for the structure (file stem, or caller-supplied).
        formation_energy: Predicted formation energy in eV/atom.
        formula: Reduced chemical formula of the structure.
        num_atoms: Number of atoms (graph nodes) in the structure.
        num_edges: Number of bonds (graph edges) in the constructed graph.
        source_path: Path the structure was read from, if any.
        metadata: Provenance of the prediction (checkpoint, device, graph config).
    """

    sample_id: str
    formation_energy: float
    formula: str
    num_atoms: int
    num_edges: int
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns the result as a JSON-serialisable dictionary."""
        return {
            "sample_id": self.sample_id,
            "formation_energy_ev_per_atom": self.formation_energy,
            "formula": self.formula,
            "num_atoms": self.num_atoms,
            "num_edges": self.num_edges,
            "source_path": self.source_path,
            "metadata": self.metadata,
        }


@dataclass
class BatchPredictionResult:
    """Outcome of a batch prediction run.

    Attributes:
        predictions: Successful predictions, in input order.
        failures: ``(sample_id, reason)`` pairs for structures that were
            skipped. A failure never aborts the batch (PRD 91).
    """

    predictions: List[PredictionResult] = field(default_factory=list)
    failures: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def num_attempted(self) -> int:
        """Total number of structures the batch attempted to predict."""
        return len(self.predictions) + len(self.failures)

    def summary(self) -> Dict[str, Any]:
        """Returns aggregate statistics for the run."""
        energies = [p.formation_energy for p in self.predictions]
        stats: Dict[str, Any] = {
            "num_attempted": self.num_attempted,
            "num_succeeded": len(self.predictions),
            "num_failed": len(self.failures),
        }
        if energies:
            tensor = torch.tensor(energies, dtype=torch.float64)
            stats.update(
                {
                    "mean_formation_energy_ev_per_atom": float(tensor.mean()),
                    "min_formation_energy_ev_per_atom": float(tensor.min()),
                    "max_formation_energy_ev_per_atom": float(tensor.max()),
                    "std_formation_energy_ev_per_atom": (
                        float(tensor.std(unbiased=True)) if len(energies) > 1 else 0.0
                    ),
                }
            )
        return stats


class Predictor:
    """Predicts formation energy for crystal structures outside the dataset.

    Example:
        >>> predictor = Predictor("experiments/checkpoints/best.ckpt")
        >>> result = predictor.predict_cif("structures/NaCl.cif")
        >>> result.formation_energy
        -2.01
    """

    def __init__(
        self,
        checkpoint_path: str,
        dataset_config_path: str = "configs/dataset.yaml",
        model_config_path: str = "configs/model.yaml",
        device: Optional[str] = None,
    ) -> None:
        """Initialises the predictor.

        Args:
            checkpoint_path: Path to the trained Lightning ``.ckpt`` checkpoint.
            dataset_config_path: Path to the dataset YAML configuration. The
                ``graph`` section must match the one used for training.
            model_config_path: Path to the model YAML configuration.
            device: Optional compute device ('cpu', 'cuda', ...). Defaults to
                'cuda' when available, else 'cpu'.
        """
        self.checkpoint_path = checkpoint_path
        self.dataset_config_path = dataset_config_path
        self.model_config_path = model_config_path

        self.dataset_config: DatasetConfig = load_dataset_config(dataset_config_path)
        self.model_config: CGCNNConfig = load_cgcnn_config(model_config_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # The builder is the single source of preprocessing truth shared with
        # training (PRD 90): same class, same GraphConfig.
        self.builder = CrystalGraphBuilder(self.graph_config)
        self._model: Optional[CGCNNLightningModule] = None

    @property
    def graph_config(self) -> GraphConfig:
        """Graph construction settings, identical to those used in training."""
        return self.dataset_config.graph

    # -- Model ---------------------------------------------------------------

    def load_model(self) -> CGCNNLightningModule:
        """Loads the checkpoint in frozen evaluation mode.

        The model is cached after the first call.

        Returns:
            CGCNNLightningModule: Model ready for inference.

        Raises:
            ModelLoadError: If the checkpoint is missing or cannot be loaded.
        """
        if self._model is not None:
            return self._model

        if not os.path.exists(self.checkpoint_path):
            raise ModelLoadError(self.checkpoint_path, "file does not exist")

        logger.info(f"Loading checkpoint from {self.checkpoint_path}...")
        try:
            with checkpoint_safe_globals():
                model = CGCNNLightningModule.load_from_checkpoint(
                    self.checkpoint_path,
                    config=self.model_config,
                    map_location=self.device,
                )
        except Exception as exc:
            raise ModelLoadError(self.checkpoint_path, str(exc)) from exc

        model.to(self.device)
        model.eval()
        model.freeze()
        self._model = model
        return model

    # -- Structure loading ---------------------------------------------------

    @staticmethod
    def load_structure(path: str) -> Structure:
        """Parses a crystal structure file (CIF) into a pymatgen Structure.

        Args:
            path: Path to the structure file.

        Returns:
            Structure: The parsed structure.

        Raises:
            StructureLoadError: If the file is missing, empty or unparseable.
        """
        if not os.path.exists(path):
            raise StructureLoadError(path, "file does not exist")
        try:
            structure = Structure.from_file(path)
        except Exception as exc:
            raise StructureLoadError(path, str(exc)) from exc

        if len(structure) == 0:
            raise StructureLoadError(path, "structure contains no sites")
        return structure

    # -- Graph construction --------------------------------------------------

    def build_graph(self, structure: Structure) -> Data:
        """Builds the inference graph for a structure.

        Delegates to the training ``CrystalGraphBuilder`` so that the tensors
        are bit-identical to those produced by the training pipeline for the
        same structure (PRD 90). The target is a sentinel, since the true
        formation energy is unknown at inference time.

        Args:
            structure: The pymatgen Structure to convert.

        Returns:
            Data: PyTorch Geometric graph for the structure.
        """
        return self.builder.build(
            structure=structure,
            record_id=_INFERENCE_RECORD_ID,
            fingerprint=_INFERENCE_FINGERPRINT,
            target=_INFERENCE_TARGET,
        )

    # -- Prediction ----------------------------------------------------------

    def _metadata(self) -> Dict[str, Any]:
        """Returns provenance metadata attached to every prediction."""
        graph = self.graph_config
        return {
            "checkpoint": self.checkpoint_path,
            "device": str(self.device),
            "units": "eV/atom",
            "graph_config": {
                "cutoff_radius": graph.cutoff_radius,
                "max_neighbors": graph.max_neighbors,
                "rbf_distance_min": graph.rbf_distance_min,
                "rbf_distance_max": graph.rbf_distance_max,
                "rbf_step_size": graph.rbf_step_size,
            },
        }

    def _forward(self, graphs: Sequence[Data]) -> torch.Tensor:
        """Runs a frozen forward pass over one or more graphs.

        Args:
            graphs: Graphs to score. All are evaluated as one PyG batch.

        Returns:
            torch.Tensor: 1D tensor of predictions, one per graph, on the CPU.
        """
        model = self.load_model()
        batch = Batch.from_data_list(list(graphs)).to(self.device)
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        predictions: torch.Tensor = out.view(-1).detach().cpu().float()
        return predictions

    def predict_structure(
        self,
        structure: Structure,
        sample_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> PredictionResult:
        """Predicts the formation energy of a single structure.

        Args:
            structure: The pymatgen Structure to predict.
            sample_id: Optional identifier. Defaults to the reduced formula.
            source_path: Optional originating file path, recorded on the result.

        Returns:
            PredictionResult: The prediction in eV/atom.

        Raises:
            PredictionError: If graph construction or the forward pass fails.
        """
        resolved_id = sample_id or structure.composition.reduced_formula
        try:
            graph = self.build_graph(structure)
            energy = float(self._forward([graph])[0])
        except Exception as exc:
            raise PredictionError(resolved_id, str(exc)) from exc

        return PredictionResult(
            sample_id=resolved_id,
            formation_energy=energy,
            formula=structure.composition.reduced_formula,
            num_atoms=int(graph.num_nodes),
            num_edges=int(graph.num_edges),
            source_path=source_path,
            metadata=self._metadata(),
        )

    def predict_cif(
        self, path: str, sample_id: Optional[str] = None
    ) -> PredictionResult:
        """Predicts the formation energy of a structure stored on disk.

        Args:
            path: Path to the CIF file.
            sample_id: Optional identifier. Defaults to the file stem.

        Returns:
            PredictionResult: The prediction in eV/atom.

        Raises:
            StructureLoadError: If the file cannot be parsed.
            PredictionError: If prediction fails.
        """
        structure = self.load_structure(path)
        resolved_id = sample_id or os.path.splitext(os.path.basename(path))[0]
        return self.predict_structure(
            structure, sample_id=resolved_id, source_path=path
        )

    # -- Batch prediction ----------------------------------------------------

    @staticmethod
    def discover_structures(directory: str) -> List[str]:
        """Lists supported structure files in a directory, sorted by name.

        Args:
            directory: Directory to scan (non-recursively).

        Returns:
            List[str]: Sorted paths of supported structure files.

        Raises:
            StructureLoadError: If the directory does not exist.
        """
        if not os.path.isdir(directory):
            raise StructureLoadError(directory, "directory does not exist")
        return [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.lower().endswith(SUPPORTED_EXTENSIONS)
        ]

    def predict_batch(
        self,
        source: Union[str, Sequence[str]],
        batch_size: Optional[int] = None,
    ) -> BatchPredictionResult:
        """Predicts formation energies for many structures.

        A structure that fails to parse or predict is logged and skipped; the
        batch always runs to completion (PRD 91).

        Args:
            source: A directory of structure files, or an explicit sequence of
                file paths.
            batch_size: Graphs per forward pass. Defaults to the dataset
                config's ``dataloader.batch_size``.

        Returns:
            BatchPredictionResult: Successful predictions and skipped structures.
        """
        paths = (
            self.discover_structures(source)
            if isinstance(source, str)
            else list(source)
        )
        size = max(1, batch_size or self.dataset_config.dataloader.batch_size)

        result = BatchPredictionResult()
        if not paths:
            logger.warning("No supported structure files found to predict.")
            return result

        logger.info(f"Predicting {len(paths)} structures (batch size {size})...")

        # Stage 1: parse and build graphs, isolating per-structure failures.
        pending: List[Tuple[str, str, Structure, Data]] = []
        for path in paths:
            sample_id = os.path.splitext(os.path.basename(path))[0]
            try:
                structure = self.load_structure(path)
                graph = self.build_graph(structure)
            except Exception as exc:
                logger.warning(f"Skipping '{sample_id}': {exc}")
                result.failures.append((sample_id, str(exc)))
                continue
            pending.append((sample_id, path, structure, graph))

        # Stage 2: score in batches, falling back to one-by-one so that a
        # single bad graph cannot discard an entire batch.
        metadata = self._metadata()
        for start in range(0, len(pending), size):
            chunk = pending[start : start + size]
            energies: Optional[torch.Tensor]
            try:
                energies = self._forward([item[3] for item in chunk])
            except Exception as exc:
                logger.warning(
                    f"Batched forward pass failed ({exc}); "
                    "retrying structures individually."
                )
                energies = None

            if energies is None:
                for sample_id, path, structure, graph in chunk:
                    try:
                        single = self._forward([graph])
                    except Exception as exc:
                        logger.warning(f"Skipping '{sample_id}': {exc}")
                        result.failures.append((sample_id, str(exc)))
                        continue
                    result.predictions.append(
                        self._make_result(
                            sample_id,
                            path,
                            structure,
                            graph,
                            float(single[0]),
                            metadata,
                        )
                    )
                continue

            for (sample_id, path, structure, graph), energy in zip(chunk, energies):
                result.predictions.append(
                    self._make_result(
                        sample_id,
                        path,
                        structure,
                        graph,
                        float(energy),
                        metadata,
                    )
                )

        logger.info(
            f"Batch prediction complete: {len(result.predictions)} succeeded, "
            f"{len(result.failures)} skipped."
        )
        return result

    @staticmethod
    def _make_result(
        sample_id: str,
        path: str,
        structure: Structure,
        graph: Data,
        energy: float,
        metadata: Dict[str, Any],
    ) -> PredictionResult:
        """Assembles a PredictionResult from a scored graph."""
        return PredictionResult(
            sample_id=sample_id,
            formation_energy=energy,
            formula=structure.composition.reduced_formula,
            num_atoms=int(graph.num_nodes),
            num_edges=int(graph.num_edges),
            source_path=path,
            metadata=metadata,
        )
