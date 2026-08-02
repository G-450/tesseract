import torch
from pymatgen.core.structure import Structure
from torch_geometric.data import Data  # type: ignore

from src.config.dataset_config import GraphConfig
from src.graphs.features import GaussianDistance


class CrystalGraphBuilder:
    """
    Constructs a PyTorch Geometric Data object from a pymatgen Structure.
    """

    def __init__(self, config: GraphConfig):
        self.config = config
        self.distance_expansion = GaussianDistance(
            dmin=config.rbf_distance_min,
            dmax=config.rbf_distance_max,
            step=config.rbf_step_size,
        )

    def build(
        self, structure: Structure, record_id: int, fingerprint: str, target: float
    ) -> Data:
        """
        Builds the crystal graph for a given structure.

        Args:
            structure (Structure): The pymatgen Structure object.
            record_id (int): The dataset record ID.
            fingerprint (str): The dataset SHA-256 fingerprint.
            target (float): The formation energy (target value).

        Returns:
            Data: The PyTorch Geometric Data object.
        """
        # Node features (atomic numbers)
        # Using atomic numbers as integer features for an embedding layer in the CGCNN
        atomic_numbers = [site.specie.Z for site in structure]
        x = torch.tensor(atomic_numbers, dtype=torch.long)

        # Get all neighbors within the cutoff radius
        all_neighbors = structure.get_all_neighbors(self.config.cutoff_radius)

        src_indices = []
        dst_indices = []
        distances = []
        image_offsets = []

        for i, site_neighbors in enumerate(all_neighbors):
            # Deterministic sorting: distance, then target index, then image offsets
            # site_neighbors is a list of PeriodicSite objects

            # Extract relevant neighbor data
            neighbor_data = []
            for neighbor in site_neighbors:
                distance = neighbor.nn_distance
                dst_idx = neighbor.index
                image = tuple(neighbor.image)

                # Exclude self-loops at the exact same periodic image (0, 0, 0)
                if i == dst_idx and image == (0, 0, 0):
                    continue

                neighbor_data.append((distance, dst_idx, image))

            # Sort deterministically
            neighbor_data.sort(key=lambda x: (x[0], x[1], x[2]))

            # Truncate to max_neighbors
            if len(neighbor_data) > self.config.max_neighbors:
                neighbor_data = neighbor_data[: self.config.max_neighbors]

            for distance, dst_idx, image in neighbor_data:
                src_indices.append(i)
                dst_indices.append(dst_idx)
                distances.append(distance)
                image_offsets.append(image)

        if len(src_indices) == 0:
            # Handle edge case with no neighbors
            # (e.g., very sparse structure or tiny cutoff)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(
                (0, len(self.distance_expansion.filter)), dtype=torch.float32
            )
            edge_image = torch.empty((0, 3), dtype=torch.long)
        else:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            distances_tensor = torch.tensor(distances, dtype=torch.float32)
            edge_attr = self.distance_expansion(distances_tensor)
            edge_image = torch.tensor(image_offsets, dtype=torch.long)

        # Build PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([target], dtype=torch.float32),
            edge_image=edge_image,
        )

        # Attach metadata
        data.dataset_record_id = record_id
        data.dataset_fingerprint = fingerprint
        data.schema_version = "1.0"

        # Validate basic integrity
        self._validate_graph(data, len(structure))

        return data

    def _validate_graph(self, data: Data, expected_nodes: int) -> None:
        """Validates graph integrity."""
        if data.x.size(0) != expected_nodes:
            raise ValueError(
                f"Graph has {data.x.size(0)} nodes, expected {expected_nodes}."
            )
        if data.edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, num_edges).")

        num_edges = data.edge_index.size(1)
        if data.edge_attr.size(0) != num_edges:
            raise ValueError("edge_attr must have the same number of rows as edges.")
        if data.edge_image.size(0) != num_edges:
            raise ValueError("edge_image must have the same number of rows as edges.")
        if (
            torch.isnan(data.x).any()
            or torch.isnan(data.edge_attr).any()
            or torch.isnan(data.y).any()
        ):
            raise ValueError("Graph contains NaN values.")
        if (
            torch.isinf(data.x).any()
            or torch.isinf(data.edge_attr).any()
            or torch.isinf(data.y).any()
        ):
            raise ValueError("Graph contains infinite values.")
