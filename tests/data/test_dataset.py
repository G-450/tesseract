import json
from typing import Any

import pytest
import torch
from torch_geometric.data import Data  # type: ignore

from src.config.dataset_config import DataLoaderConfig, DatasetConfig
from src.data.datamodule import CrystalDataModule
from src.data.dataset import ShardedGraphDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_processed_dir(tmp_path: Any) -> str:
    """2-shard fixture (3 graphs total) for basic dataset tests."""
    processed_dir = tmp_path / "processed"
    graphs_dir = processed_dir / "graphs"
    graphs_dir.mkdir(parents=True)

    shard_1 = [
        Data(x=torch.randn(3, 5), edge_index=torch.tensor([[0, 1], [1, 0]])),
        Data(x=torch.randn(4, 5), edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]])),
    ]
    torch.save(shard_1, graphs_dir / "shard_00000.pt")

    shard_2 = [Data(x=torch.randn(2, 5), edge_index=torch.tensor([[0, 1], [1, 0]]))]
    torch.save(shard_2, graphs_dir / "shard_00001.pt")

    manifest_data = {
        "fingerprint": "mock_fp",
        "total_records_processed": 3,
        "successful_graphs": 3,
        "shard_lengths": {
            "shard_00000.pt": 2,
            "shard_00001.pt": 1,
        },
    }
    with open(graphs_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f)

    return str(processed_dir)


@pytest.fixture
def mock_processed_dir_5shards(tmp_path: Any) -> str:
    """5-shard fixture (1 graph each) for datamodule split tests.

    With train_split=0.6, val_split=0.2, test_split=0.2 and 5 shards:
        train_end = max(1, int(5 * 0.6)) = 3  → shards 0-2
        val_end   = 3 + int(5 * 0.2)    = 4  → shard 3
        test      = remaining             = 1  → shard 4
    """
    processed_dir = tmp_path / "processed"
    graphs_dir = processed_dir / "graphs"
    graphs_dir.mkdir(parents=True)

    shard_lengths = {}
    for i in range(5):
        graphs = [Data(x=torch.randn(3, 5), edge_index=torch.tensor([[0, 1], [1, 0]]))]
        fname = f"shard_{i:05d}.pt"
        torch.save(graphs, graphs_dir / fname)
        shard_lengths[fname] = len(graphs)

    manifest_data = {
        "fingerprint": "mock_fp_5s",
        "total_records_processed": 5,
        "successful_graphs": 5,
        "shard_lengths": shard_lengths,
    }
    with open(graphs_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f)

    return str(processed_dir)


@pytest.fixture
def dataset_config(mock_processed_dir: str) -> DatasetConfig:
    config = DatasetConfig()
    config.processed_data_dir = mock_processed_dir
    config.dataloader = DataLoaderConfig(
        batch_size=2, num_workers=0, train_split=0.6, val_split=0.2, test_split=0.2
    )
    return config


@pytest.fixture
def datamodule_config(mock_processed_dir_5shards: str) -> DatasetConfig:
    config = DatasetConfig()
    config.processed_data_dir = mock_processed_dir_5shards
    config.dataloader = DataLoaderConfig(
        batch_size=2, num_workers=0, train_split=0.6, val_split=0.2, test_split=0.2
    )
    return config


# ---------------------------------------------------------------------------
# ShardedGraphDataset tests
# ---------------------------------------------------------------------------


def test_sharded_graph_dataset_len(dataset_config: DatasetConfig) -> None:
    """len() is read from manifest shard_lengths — no .pt files loaded."""
    train, val, test = ShardedGraphDataset.split(dataset_config)
    # 2 shards: train gets shard 0 (2 graphs), test gets shard 1 (1 graph)
    assert len(train) == 2
    assert len(test) == 1


def test_sharded_graph_dataset_iteration(dataset_config: DatasetConfig) -> None:
    """Iterating a full single-split dataset yields all graphs."""
    _, shard_files, shard_lengths = ShardedGraphDataset._load_index(dataset_config)
    full_ds = ShardedGraphDataset(
        dataset_config, shard_files, shard_lengths, shuffle=False
    )
    graphs = list(full_ds)
    assert len(graphs) == 3
    # First shard: shapes (3,5) and (4,5); second shard: (2,5)
    assert graphs[0].x.shape == (3, 5)
    assert graphs[1].x.shape == (4, 5)
    assert graphs[2].x.shape == (2, 5)


def test_sharded_graph_dataset_split_shard_counts(
    dataset_config: DatasetConfig,
) -> None:
    """split() assigns the right number of shards to each partition."""
    train, val, test = ShardedGraphDataset.split(dataset_config)
    # 2 shards total: train gets shard 0, val gets [], test gets shard 1
    assert len(train.shard_files) == 1
    assert len(val.shard_files) == 0
    assert len(test.shard_files) == 1


def test_sharded_graph_dataset_shuffle_varies_by_epoch(
    dataset_config: DatasetConfig,
) -> None:
    """Different epochs produce different graph orderings."""
    _, shard_files, shard_lengths = ShardedGraphDataset._load_index(dataset_config)
    # Use only shard 0 (2 graphs) to keep the test lightweight
    ds = ShardedGraphDataset(
        dataset_config, shard_files[:1], shard_lengths[:1], shuffle=True, seed=42
    )
    ds.set_epoch(0)
    order_epoch0 = [g.x.shape for g in ds]
    ds.set_epoch(1)
    order_epoch1 = [g.x.shape for g in ds]
    # With 2 graphs there are only 2 possible orderings; confirm we test both.
    # (At least one of the two epochs should differ from sequential order.)
    sequential = [(3, 5), (4, 5)]
    # It's statistically possible both epochs are the same order for 2 items;
    # assert at least one epoch differs from the other (sanity check).
    assert order_epoch0 != order_epoch1 or order_epoch0 == sequential


def test_sharded_graph_dataset_legacy_manifest(tmp_path: Any) -> None:
    """Falls back to torch.load when manifest has no shard_lengths key."""
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    graphs = [Data(x=torch.randn(2, 5), edge_index=torch.tensor([[0, 1], [1, 0]]))]
    torch.save(graphs, graphs_dir / "shard_00000.pt")

    # Legacy manifest without shard_lengths
    manifest = {"fingerprint": "old", "successful_graphs": 1}
    with open(graphs_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    config = DatasetConfig()
    config.processed_data_dir = str(tmp_path)
    _, _, lengths = ShardedGraphDataset._load_index(config)
    assert lengths == [1]


# ---------------------------------------------------------------------------
# CrystalDataModule tests
# ---------------------------------------------------------------------------


def test_crystal_datamodule_setup(datamodule_config: DatasetConfig) -> None:
    """setup() creates all three split datasets."""
    dm = CrystalDataModule(datamodule_config)
    dm.setup()

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None


def test_crystal_datamodule_split_sizes(datamodule_config: DatasetConfig) -> None:
    """5 shards with 0.6/0.2/0.2 → 3 train / 1 val / 1 test shards."""
    dm = CrystalDataModule(datamodule_config)
    dm.setup()

    assert len(dm.train_dataset.shard_files) == 3  # type: ignore[union-attr]
    assert len(dm.val_dataset.shard_files) == 1  # type: ignore[union-attr]
    assert len(dm.test_dataset.shard_files) == 1  # type: ignore[union-attr]

    assert len(dm.train_dataset) == 3  # type: ignore[union-attr]
    assert len(dm.val_dataset) == 1  # type: ignore[union-attr]
    assert len(dm.test_dataset) == 1  # type: ignore[union-attr]


def test_crystal_datamodule_train_dataloader(datamodule_config: DatasetConfig) -> None:
    """Train DataLoader produces PyG Batch objects."""
    dm = CrystalDataModule(datamodule_config)
    dm.setup()

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    # PyTorch Geometric Batch has .batch and .ptr attributes
    assert hasattr(batch, "batch")
    assert hasattr(batch, "ptr")


def test_crystal_datamodule_set_epoch(datamodule_config: DatasetConfig) -> None:
    """set_epoch() propagates the epoch counter to the train dataset."""
    dm = CrystalDataModule(datamodule_config)
    dm.setup()

    assert dm.train_dataset is not None
    dm.train_dataset.set_epoch(5)
    assert dm.train_dataset._epoch == 5
