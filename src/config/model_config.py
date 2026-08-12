import os
from dataclasses import dataclass, field
import yaml  # type: ignore

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
    if not os.path.exists(config_path):
        return CGCNNConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return CGCNNConfig()

    model_data = data.get("model", {})
    optimizer_data = data.get("optimizer", {})

    model_config = ModelConfig(**model_data) if model_data else ModelConfig()
    optimizer_config = OptimizerConfig(**optimizer_data) if optimizer_data else OptimizerConfig()

    return CGCNNConfig(model=model_config, optimizer=optimizer_config)
