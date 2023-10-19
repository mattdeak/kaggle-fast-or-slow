from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_add_pool

INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24


class SAGEBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        with_residual: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv = SAGEConv(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.with_residual = with_residual
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        f = self.conv(x, edge_index)
        f = cast(torch.Tensor, F.gelu(f))
        f = self.norm(f)
        f = self.dropout(f)

        if self.with_residual:
            f += x

        return f


class LinearBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        with_residual: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.with_residual = with_residual
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.linear(x)
        f = cast(torch.Tensor, F.gelu(f))
        f = self.norm(f)
        f = self.dropout(f)

        if self.with_residual:
            f += x
        return f


class SAGEMLP(nn.Module):
    def __init__(
        self,
        graph_input_dim: int = INPUT_DIM,
        sage_channels: int = 64,
        sage_layers: int = 6,
        linear_channels: int = 32,
        linear_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gcns = nn.ModuleList(
            [
                SAGEBlock(
                    graph_input_dim, sage_channels, with_residual=False, dropout=dropout
                ),
            ]
            + [
                SAGEBlock(sage_channels, sage_channels, dropout=dropout)
                for _ in range(sage_layers)
            ]
        )

        self.mlp = nn.Sequential(
            LinearBlock(
                sage_channels,
                linear_channels,
                with_residual=False,
                dropout=dropout,
            ),
            *[
                LinearBlock(linear_channels, linear_channels, dropout=dropout)
                for _ in range(linear_layers)
            ],
            nn.Linear(linear_channels, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x: torch.Tensor = data.x
        batch: torch.Tensor = data.batch
        edge_index: torch.Tensor = data.edge_index

        for gcn_block in self.gcns:
            x = gcn_block(x, edge_index)

        sum_pool = global_add_pool(x, batch)
        return self.mlp(sum_pool)
