"""Dataset configuration: paths, graph parameters and loader settings."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml  # type: ignore

from src.config.exceptions import ConfigFileError, UnknownConfigKeyError
from src.config.experiment_config import _build


@dataclass
class GraphConfig:
    cutoff_radius: float = 8.0
    max_neighbors: int = 12
    rbf_distance_min: float = 0.0
    rbf_distance_max: float = 8.0
    rbf_step_size: float = 0.2
    shard_size: int = 1000


@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class DatasetConfig:
    raw_data_path: str = "datasets/raw/mp_e_form.json"
    metadata_dir: str = "datasets/metadata"
    processed_data_dir: str = "datasets/processed"
    require_structure: bool = True
    require_target: bool = True
    target_column: str = "e_form"
    graph: GraphConfig = field(default_factory=GraphConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


def load_dataset_config(config_path: str = "configs/dataset.yaml") -> DatasetConfig:
    """Loads the dataset configuration from a YAML file.

    Unknown keys raise ``UnknownConfigKeyError`` rather than a raw
    ``TypeError`` (PRD §122).

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        DatasetConfig: The loaded configuration object.

    Raises:
        ConfigFileError: If the file cannot be read or parsed.
        UnknownConfigKeyError: If the file contains unrecognised keys.
    """
    if not os.path.exists(config_path):
        return DatasetConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigFileError(config_path, str(exc)) from exc

    if not data:
        return DatasetConfig()
    if not isinstance(data, dict):
        raise ConfigFileError(config_path, "top-level document is not a mapping")

    unknown_top = set(data) - {"dataset"}
    if unknown_top:
        raise UnknownConfigKeyError("<root>", list(unknown_top), ["dataset"])

    dataset_data: Dict[str, Any] = dict(data.get("dataset") or {})

    graph_data = dict(dataset_data.pop("graph", None) or {})
    graph_config = _build(GraphConfig, graph_data, "dataset.graph")

    dataloader_data = dict(dataset_data.pop("dataloader", None) or {})
    dataloader_config = _build(DataLoaderConfig, dataloader_data, "dataset.dataloader")

    return _build(
        DatasetConfig,
        {**dataset_data, "graph": graph_config, "dataloader": dataloader_config},
        "dataset",
    )
