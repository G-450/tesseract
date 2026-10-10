"""Model and optimizer configuration for CGCNN."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml  # type: ignore

from src.config.exceptions import ConfigFileError, UnknownConfigKeyError
from src.config.experiment_config import _build


@dataclass
class OptimizerConfig:
    lr: float = 0.001
    weight_decay: float = 0.0
    lr_patience: int = 10
    lr_factor: float = 0.5


@dataclass
class ModelConfig:
    node_dim: int = 92
    edge_dim: int = 41
    hidden_dim: int = 64
    num_conv_layers: int = 3
    pooling: str = "sum"


@dataclass
class CGCNNConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


def load_cgcnn_config(config_path: str = "configs/model.yaml") -> CGCNNConfig:
    """Loads the model configuration from a YAML file.

    Unknown keys raise ``UnknownConfigKeyError`` rather than a raw
    ``TypeError`` (PRD §122).

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        CGCNNConfig: The loaded configuration object.

    Raises:
        ConfigFileError: If the file cannot be read or parsed.
        UnknownConfigKeyError: If the file contains unrecognised keys.
    """
    if not os.path.exists(config_path):
        return CGCNNConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigFileError(config_path, str(exc)) from exc

    if not data:
        return CGCNNConfig()
    if not isinstance(data, dict):
        raise ConfigFileError(config_path, "top-level document is not a mapping")

    allowed_top = {"model", "optimizer"}
    unknown_top = set(data) - allowed_top
    if unknown_top:
        raise UnknownConfigKeyError("<root>", list(unknown_top), list(allowed_top))

    model_data: Dict[str, Any] = dict(data.get("model") or {})
    optimizer_data: Dict[str, Any] = dict(data.get("optimizer") or {})

    model_config = _build(ModelConfig, model_data, "model")
    optimizer_config = _build(OptimizerConfig, optimizer_data, "optimizer")

    return CGCNNConfig(model=model_config, optimizer=optimizer_config)
