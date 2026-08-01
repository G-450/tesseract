import os
from dataclasses import dataclass

import yaml


@dataclass
class DatasetConfig:
    raw_data_path: str = "datasets/raw/mp_e_form.json"
    metadata_dir: str = "datasets/metadata"
    processed_data_dir: str = "datasets/processed"
    require_structure: bool = True
    require_target: bool = True
    target_column: str = "e_form"


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
        return DatasetConfig(**data["dataset"])
    return DatasetConfig()
