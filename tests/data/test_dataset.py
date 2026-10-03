import json
import os
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


def _write_manifest(graphs_dir: Any, shard_lengths: dict) -> None:
    manifest_data = {
        "fingerprint": "mock_fp",
        "total_records_processed": sum(shard_lengths.values()),
        "successful_graphs": sum(shard_lengths.values()),
        "shard_lengths": shard_lengths,
    }
    with open(graphs_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f)


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

    _write_manifest(graphs_dir, {"shard_00000.pt": 2, "shard_00001.pt": 1})
    return str(processed_dir)


@pytest.fixture
def mock_processed_dir_5graphs(tmp_path: Any) -> str:
    """5-shard fixture (1 graph each) for datamodule split tests.

    With train_split=0.6, val_split=0.2, test_split=0.2 and 5 graphs:
        train = int(5 * 0.6) = 3, val = int(5 * 0.2) = 1, test = 5 - 3 - 1 = 1
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

    _write_manifest(graphs_dir, shard_lengths)
    return str(processed_dir)


def _make_config(processed_dir: str) -> DatasetConfig:
    config = DatasetConfig()
    config.processed_data_dir = processed_dir
    config.dataloader = DataLoaderConfig(
        batch_size=2, num_workers=0, train_split=0.6, val_split=0.2, test_split=0.2
    )
    return config


@pytest.fixture
def dataset_config(mock_processed_dir: str) -> DatasetConfig:
    return _make_config(mock_processed_dir)


@pytest.fixture
def datamodule_config(mock_processed_dir_5graphs: str) -> DatasetConfig:
    return _make_config(mock_processed_dir_5graphs)


# ---------------------------------------------------------------------------
# ShardedGraphDataset tests
# ---------------------------------------------------------------------------


def test_sharded_graph_dataset(dataset_config: DatasetConfig) -> None:
    """All graphs from all shards are loaded, in shard order."""
    dataset = ShardedGraphDataset(dataset_config)

    # Total graphs should be 2 from shard_00000 + 1 from shard_00001
    assert len(dataset) == 3
    assert dataset[0].x.shape == (3, 5)
    assert dataset[1].x.shape == (4, 5)
    assert dataset[2].x.shape == (2, 5)


def test_sharded_graph_dataset_reuses_compiled(
    dataset_config: DatasetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh compiled dataset is loaded directly without re-collating shards."""
    ShardedGraphDataset(dataset_config)
    compiled_path = os.path.join(
        dataset_config.processed_data_dir, "graphs", "compiled_dataset.pt"
    )
    assert os.path.exists(compiled_path)

    def fail_compile(self: ShardedGraphDataset) -> None:
        raise AssertionError("compiled dataset should have been reused")

    monkeypatch.setattr(ShardedGraphDataset, "_compile_shards", fail_compile)
    assert len(ShardedGraphDataset(dataset_config)) == 3


def test_sharded_graph_dataset_recompiles_when_manifest_newer(
    dataset_config: DatasetConfig,
) -> None:
    """Re-running the pipeline (newer manifest) invalidates the compiled cache."""
    ShardedGraphDataset(dataset_config)
    graphs_dir = os.path.join(dataset_config.processed_data_dir, "graphs")

    # Simulate a pipeline re-run that adds a third shard
    extra = [Data(x=torch.randn(6, 5), edge_index=torch.tensor([[0, 1], [1, 0]]))]
    torch.save(extra, os.path.join(graphs_dir, "shard_00002.pt"))
    compiled_mtime = os.path.getmtime(os.path.join(graphs_dir, "compiled_dataset.pt"))
    manifest_path = os.path.join(graphs_dir, "manifest.json")
    os.utime(manifest_path, (compiled_mtime + 10, compiled_mtime + 10))

    dataset = ShardedGraphDataset(dataset_config)
    assert len(dataset) == 4
    assert dataset[3].x.shape == (6, 5)


def test_sharded_graph_dataset_corrupt_shard_raises(
    dataset_config: DatasetConfig,
) -> None:
    """A shard that fails to load aborts compilation instead of dropping data."""
    graphs_dir = os.path.join(dataset_config.processed_data_dir, "graphs")
    with open(os.path.join(graphs_dir, "shard_00001.pt"), "wb") as f:
        f.write(b"not a torch file")

    with pytest.raises(Exception):
        ShardedGraphDataset(dataset_config)
    assert not os.path.exists(os.path.join(graphs_dir, "compiled_dataset.pt"))


def test_sharded_graph_dataset_missing_manifest(tmp_path: Any) -> None:
    (tmp_path / "graphs").mkdir()
    with pytest.raises(FileNotFoundError):
        ShardedGraphDataset(_make_config(str(tmp_path)))


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
    """5 graphs with 0.6/0.2/0.2 → 3 train / 1 val / 1 test, disjoint."""
    dm = CrystalDataModule(datamodule_config)
    dm.setup()

    assert len(dm.train_dataset) == 3  # type: ignore[arg-type]
    assert len(dm.val_dataset) == 1  # type: ignore[arg-type]
    assert len(dm.test_dataset) == 1  # type: ignore[arg-type]

    indices = (
        list(dm.train_dataset.indices)  # type: ignore[union-attr]
        + list(dm.val_dataset.indices)  # type: ignore[union-attr]
        + list(dm.test_dataset.indices)  # type: ignore[union-attr]
    )
    assert sorted(indices) == [0, 1, 2, 3, 4]


def test_crystal_datamodule_split_is_deterministic(
    datamodule_config: DatasetConfig,
) -> None:
    """The fixed seed yields the same split across runs."""
    dm1 = CrystalDataModule(datamodule_config)
    dm1.setup()
    dm2 = CrystalDataModule(datamodule_config)
    dm2.setup()

    assert list(dm1.test_dataset.indices) == list(  # type: ignore[union-attr]
        dm2.test_dataset.indices  # type: ignore[union-attr]
    )


def test_crystal_datamodule_train_dataloader(datamodule_config: DatasetConfig) -> None:
    """Train DataLoader produces PyG Batch objects."""
    dm = CrystalDataModule(datamodule_config)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    # PyTorch Geometric Batch has .batch and .ptr attributes
    assert hasattr(batch, "batch")
    assert hasattr(batch, "ptr")
