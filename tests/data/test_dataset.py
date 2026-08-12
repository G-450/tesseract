import json

import pytest
import torch
from torch_geometric.data import Data

from src.config.dataset_config import DataLoaderConfig, DatasetConfig
from src.data.datamodule import CrystalDataModule
from src.data.dataset import ShardedGraphDataset


@pytest.fixture
def mock_processed_dir(tmp_path):
    # Setup mock processed dir
    processed_dir = tmp_path / "processed"
    graphs_dir = processed_dir / "graphs"
    graphs_dir.mkdir(parents=True)

    # Create fake manifest
    manifest_data = {
        "fingerprint": "mock_fp",
        "total_records_processed": 10,
        "successful_graphs": 10,
    }
    manifest_path = graphs_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    # Create fake shards
    shard_1 = [
        Data(x=torch.randn(3, 5), edge_index=torch.tensor([[0, 1], [1, 0]])),
        Data(x=torch.randn(4, 5), edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]])),
    ]
    torch.save(shard_1, graphs_dir / "shard_00000.pt")

    shard_2 = [Data(x=torch.randn(2, 5), edge_index=torch.tensor([[0, 1], [1, 0]]))]
    torch.save(shard_2, graphs_dir / "shard_00001.pt")

    return str(processed_dir)


@pytest.fixture
def dataset_config(mock_processed_dir):
    config = DatasetConfig()
    config.processed_data_dir = mock_processed_dir
    config.dataloader = DataLoaderConfig(
        batch_size=2, num_workers=0, train_split=0.6, val_split=0.2, test_split=0.2
    )
    return config


def test_sharded_graph_dataset(dataset_config):
    dataset = ShardedGraphDataset(dataset_config)

    # Total graphs should be 2 from shard_00000 + 1 from shard_00001
    assert len(dataset) == 3

    # Retrieve first graph
    data_0 = dataset[0]
    assert data_0.x.shape == (3, 5)

    # Retrieve third graph (from second shard)
    data_2 = dataset[2]
    assert data_2.x.shape == (2, 5)


def test_crystal_datamodule(dataset_config):
    datamodule = CrystalDataModule(dataset_config)
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    # Train = 3 * 0.6 = 1, Val = 3 * 0.2 = 0, Test = 3 - 1 - 0 = 2
    # The actual random_split might floor differently depending on generator,
    # but the total sum of lengths should equal dataset length (3).
    total_len = (
        len(datamodule.train_dataset)
        + len(datamodule.val_dataset)
        + len(datamodule.test_dataset)
    )
    assert total_len == 3

    # Test dataloader batching
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # It's a PyTorch Geometric Batch object
    assert hasattr(batch, "batch")
    assert hasattr(batch, "ptr")
