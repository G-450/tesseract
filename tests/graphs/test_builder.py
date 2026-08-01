from typing import Any
import pytest
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from src.config.dataset_config import GraphConfig
from src.graphs.builder import CrystalGraphBuilder


@pytest.fixture
def simple_cubic_structure() -> Any:
    lattice = Lattice.cubic(5.0)
    return Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture
def builder() -> Any:
    config = GraphConfig(
        cutoff_radius=6.0,
        max_neighbors=12,
        rbf_distance_min=0.0,
        rbf_distance_max=6.0,
        rbf_step_size=0.2,
    )
    return CrystalGraphBuilder(config)


def test_builder_basic(builder, simple_cubic_structure) -> None:  # type: ignore
    data = builder.build(
        simple_cubic_structure, record_id=1, fingerprint="abc", target=-1.5
    )

    assert data.x.shape == (2,)
    assert data.x[0].item() == 11  # Na
    assert data.x[1].item() == 17  # Cl

    assert data.y.item() == -1.5
    assert data.dataset_record_id == 1
    assert data.dataset_fingerprint == "abc"
    assert data.schema_version == "1.0"

    assert data.edge_index.shape[0] == 2
    num_edges = data.edge_index.shape[1]
    assert num_edges > 0

    assert data.edge_attr.shape == (num_edges, 31)  # (6.0 - 0.0) / 0.2 + 1 = 31
    assert data.edge_image.shape == (num_edges, 3)


def test_builder_self_loops_excluded(builder) -> None:  # type: ignore
    # Create a very small cell where periodic self-interactions are possible
    lattice = Lattice.cubic(2.0)
    struct = Structure(lattice, ["H"], [[0.0, 0.0, 0.0]])

    data = builder.build(struct, record_id=1, fingerprint="abc", target=0.0)

    # We should have edges from H to its periodic images, but NOT to image (0,0,0)
    num_edges = data.edge_index.shape[1]
    assert num_edges > 0

    for i in range(num_edges):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        image = tuple(data.edge_image[i].tolist())

        # If it's the same atom, the image offset must NOT be (0, 0, 0)
        if src == dst:
            assert image != (0, 0, 0)


def test_builder_max_neighbors() -> None:  # type: ignore
    config = GraphConfig(cutoff_radius=10.0, max_neighbors=4)
    builder = CrystalGraphBuilder(config)

    lattice = Lattice.cubic(3.0)
    struct = Structure(lattice, ["H"], [[0.0, 0.0, 0.0]])

    data = builder.build(struct, record_id=1, fingerprint="abc", target=0.0)

    # Struct has many neighbors within 10.0 Angstroms,
    # but we truncated to max_neighbors=4
    # Since there's 1 atom, there should be exactly 4 edges total.
    assert data.edge_index.shape[1] == 4


def test_builder_deterministic_ordering(builder, simple_cubic_structure) -> None:  # type: ignore
    data1 = builder.build(
        simple_cubic_structure, record_id=1, fingerprint="abc", target=-1.5
    )
    data2 = builder.build(
        simple_cubic_structure, record_id=1, fingerprint="abc", target=-1.5
    )

    assert torch.all(data1.edge_index == data2.edge_index)
    assert torch.all(data1.edge_image == data2.edge_image)
    assert torch.allclose(data1.edge_attr, data2.edge_attr)
