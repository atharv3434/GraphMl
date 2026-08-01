"""GNN model factory: GCN, GraphSAGE, and GAT for node classification."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from graph_ml.config import ModelConfig


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        dims = [in_channels] + [hidden_dim] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        dims = [in_channels] + [hidden_dim] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(nn.Module):
    def __init__(
        self, in_channels: int, hidden_dim: int, out_channels: int,
        num_layers: int, dropout: float, heads: int,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, dropout=dropout))
        else:
            self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden_dim * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


_REGISTRY = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
}


def build_model(config: ModelConfig, in_channels: int, out_channels: int) -> nn.Module:
    if config.type not in _REGISTRY:
        raise ValueError(f"Unknown model type '{config.type}'. Available: {list(_REGISTRY.keys())}")

    if config.type == "gat":
        return GAT(
            in_channels, config.hidden_dim, out_channels,
            config.num_layers, config.dropout, config.heads,
        )
    return _REGISTRY[config.type](
        in_channels, config.hidden_dim, out_channels, config.num_layers, config.dropout
    )
