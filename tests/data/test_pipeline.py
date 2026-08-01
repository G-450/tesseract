import json

import torch

from src.config.dataset_config import DatasetConfig, GraphConfig
from src.data.pipeline import GraphPipeline


def test_pipeline_processing(tmp_path) -> None:  # type: ignore
    # Create a small dummy JSON dataset resembling the actual one
    dummy_json_path = tmp_path / "dummy_dataset.json"
    processed_dir = tmp_path / "processed"

    # We will make 2 valid records and 1 invalid record
    dummy_data = {
        "columns": ["structure", "e_form"],
        "data": [
            [
                {
                    "@module": "pymatgen.core.structure",
                    "@class": "Structure",
                    "lattice": {"matrix": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]},
                    "sites": [
                        {
                            "species": [{"element": "Na", "occu": 1}],
                            "abc": [0.0, 0.0, 0.0],
                        }
                    ],
                },
                -1.5,
            ],
            [
                {
                    "@module": "pymatgen.core.structure",
                    "@class": "Structure",
                    "lattice": {"matrix": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]},
                    "sites": [
                        {
                            "species": [{"element": "Fe", "occu": 1}],
                            "abc": [0.0, 0.0, 0.0],
                        }
                    ],
                },
                0.5,
            ],
            ["not a structure dict", 1.0],
        ],
    }

    with open(dummy_json_path, "w") as f:
        json.dump(dummy_data, f)

    config = DatasetConfig(
        raw_data_path=str(dummy_json_path),
        processed_data_dir=str(processed_dir),
        graph=GraphConfig(cutoff_radius=6.0, shard_size=1),
    )

    pipeline = GraphPipeline(config)
    report = pipeline.process()

    assert report["total_records_processed"] == 3
    assert report["successful_graphs"] == 2
    assert report["failed_graphs"] == 1

    # Since shard_size = 1 and 2 successful graphs, we should have 2 shards created
    assert report["shards_created"] == 2

    graphs_dir = processed_dir / "graphs"
    assert graphs_dir.exists()

    # Check manifest
    manifest_path = graphs_dir / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        assert manifest["successful_graphs"] == 2
        assert len(manifest["failure_details"]) == 1

    # Check shard files
    shard_0 = graphs_dir / "shard_00000.pt"
    shard_1 = graphs_dir / "shard_00001.pt"
    assert shard_0.exists()
    assert shard_1.exists()

    # Load and check graph from shard 0
    graphs_0 = torch.load(shard_0, weights_only=False)
    assert len(graphs_0) == 1
    g0 = graphs_0[0]
    assert g0.x.shape == (1,)
    assert g0.x[0].item() == 11  # Na
    assert g0.y.item() == -1.5
