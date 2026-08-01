import os
from dataclasses import dataclass, field

import yaml  # type: ignore


@dataclass
class GraphConfig:
    cutoff_radius: float = 8.0
    max_neighbors: int = 12
    rbf_distance_min: float = 0.0
    rbf_distance_max: float = 8.0
    rbf_step_size: float = 0.2
    shard_size: int = 1000


@dataclass
class DatasetConfig:
    raw_data_path: str = "datasets/raw/mp_e_form.json"
    metadata_dir: str = "datasets/metadata"
    processed_data_dir: str = "datasets/processed"
    require_structure: bool = True
    require_target: bool = True
    target_column: str = "e_form"
    graph: GraphConfig = field(default_factory=GraphConfig)


def load_dataset_config(config_path: str = "configs/dataset.yaml") -> DatasetConfig:
    """Loads the dataset configuration from a YAML file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        DatasetConfig: The loaded configuration object.
    """
    if not os.path.exists(config_path):
        return DatasetConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data and "dataset" in data:
        dataset_data = data["dataset"].copy()

        # Parse graph config if present
        graph_data = dataset_data.pop("graph", None)
        graph_config = GraphConfig(**graph_data) if graph_data else GraphConfig()

        return DatasetConfig(**dataset_data, graph=graph_config)
    return DatasetConfig()
