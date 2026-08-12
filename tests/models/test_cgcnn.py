import pytest
import torch
from torch_geometric.data import Data, Batch

from src.config.model_config import ModelConfig, CGCNNConfig
from src.models.layers import CGCNNConv
from src.models.cgcnn import CGCNN
from src.models.lightning_module import CGCNNLightningModule

@pytest.fixture
def mock_batch():
    # 2 graphs in a batch
    # Graph 1: 3 nodes, 4 edges
    x1 = torch.randn(3, 92)
    edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr1 = torch.randn(4, 41)
    y1 = torch.tensor([1.5])
    data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1, y=y1)
    
    # Graph 2: 2 nodes, 2 edges
    x2 = torch.randn(2, 92)
    edge_index2 = torch.tensor([[0, 1], [1, 0]])
    edge_attr2 = torch.randn(2, 41)
    y2 = torch.tensor([0.8])
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2, y=y2)
    
    return Batch.from_data_list([data1, data2])

@pytest.fixture
def model_config():
    return ModelConfig(
        node_dim=92,
        edge_dim=41,
        hidden_dim=32,
        num_conv_layers=2,
        pooling="sum"
    )

def test_cgcnn_conv(mock_batch):
    conv = CGCNNConv(node_dim=92, edge_dim=41)
    
    # Forward pass
    out = conv(mock_batch.x, mock_batch.edge_index, mock_batch.edge_attr)
    
    # Output should have same shape as input node features
    assert out.shape == mock_batch.x.shape
    assert out.dtype == torch.float32

def test_cgcnn_model(mock_batch, model_config):
    model = CGCNN(model_config)
    
    out = model(mock_batch.x, mock_batch.edge_index, mock_batch.edge_attr, mock_batch.batch)
    
    # Output should be [batch_size, 1]
    assert out.shape == (mock_batch.num_graphs, 1)

def test_cgcnn_lightning_module(mock_batch):
    config = CGCNNConfig()
    config.model.hidden_dim = 32 # override for faster test
    config.model.num_conv_layers = 2
    
    module = CGCNNLightningModule(config)
    
    # Test training step
    loss = module.training_step(mock_batch, 0)
    assert loss.ndim == 0 # scalar
    assert not torch.isnan(loss)
    
    # Test optimizer config
    opt_dict = module.configure_optimizers()
    assert "optimizer" in opt_dict
    assert "lr_scheduler" in opt_dict
