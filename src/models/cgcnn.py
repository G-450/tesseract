import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from src.models.layers import CGCNNConv
from src.config.model_config import ModelConfig


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network.
    """
    def __init__(self, config: ModelConfig):
        super(CGCNN, self).__init__()
        self.config = config
        
        # Initial atom embedding layer
        self.embedding = nn.Linear(config.node_dim, config.hidden_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            CGCNNConv(node_dim=config.hidden_dim, edge_dim=config.edge_dim)
            for _ in range(config.num_conv_layers)
        ])
        
        # Fully connected layers after pooling
        self.fc_pre = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.softplus = nn.Softplus()
        self.fc_out = nn.Linear(config.hidden_dim, 1)
        
        # Pooling function selection
        if config.pooling == "mean":
            self.pool = global_mean_pool
        elif config.pooling == "max":
            self.pool = global_max_pool
        elif config.pooling == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling type: {config.pooling}")

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass.
        Args:
            x (Tensor): Node features [num_nodes, node_dim]
            edge_index (LongTensor): Edge indices [2, num_edges]
            edge_attr (Tensor): Edge features [num_edges, edge_dim]
            batch (LongTensor): Batch assignment for each node [num_nodes]
            
        Returns:
            Tensor: Output scalar predictions [batch_size, 1]
        """
        # Atom embedding
        v = self.embedding(x)
        
        # Message passing layers
        for conv in self.convs:
            v = conv(v, edge_index, edge_attr)
            
        # Crystal pooling
        c = self.pool(v, batch)
        
        # Output multi-layer perceptron
        c = self.fc_pre(c)
        c = self.softplus(c)
        out = self.fc_out(c)
        
        return out
