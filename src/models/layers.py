from typing import Any

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing  # type: ignore


class CGCNNConv(MessagePassing):
    """
    Crystal Graph Convolutional Neural Network layer.
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super(CGCNNConv, self).__init__(aggr="add")
        self.n_dim = node_dim
        self.e_dim = edge_dim

        # Linear layer that takes concatenated (node_i, node_j, edge) -> 2 * node_dim
        # We output 2*node_dim to split it into the filter (sigmoid)
        # and core (softplus) components
        self.fc = nn.Linear(2 * node_dim + edge_dim, 2 * node_dim)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Any:
        """
        Args:
            x (Tensor): Node feature tensor of shape [num_nodes, node_dim]
            edge_index (LongTensor): Edge index tensor of shape [2, num_edges]
            edge_attr (Tensor): Edge feature tensor of shape [num_edges, edge_dim]
        """
        # Start propagating messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Apply batch normalization and residual connection
        out = self.bn(out)
        out = out + x
        return out

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> Any:
        """
        Message construction.
        Args:
            x_i (Tensor): Node features of the target node [num_edges, node_dim]
            x_j (Tensor): Node features of the source node [num_edges, node_dim]
            edge_attr (Tensor): Edge features [num_edges, edge_dim]
        """
        # Concatenate node_i, node_j, and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Pass through linear layer
        z = self.fc(z)

        # Split into filter and core parts
        # z_filter, z_core will each have shape [num_edges, node_dim]
        z_filter, z_core = z.chunk(2, dim=-1)

        # Apply non-linearities and element-wise multiplication
        out = self.sigmoid(z_filter) * self.softplus(z_core)

        return out
