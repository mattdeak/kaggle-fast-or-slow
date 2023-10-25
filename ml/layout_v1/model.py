from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, SAGPooling
from torch_geometric.nn.pool import global_max_pool

INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24


class SAGEBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        with_residual: bool = True,
        dropout: float = 0.5,
        output_dim: int | None = None,
        pooling_ratio: float | None = None,
    ):
        if output_dim is None and pooling_ratio is None:
            raise ValueError("Either output_dim or pooling_ratio must be provided.")
        elif output_dim is None and pooling_ratio is not None:
            output_dim = int(input_dim * pooling_ratio)
        elif output_dim is None:
            raise RuntimeError("This should never happen.")

        super().__init__()
        self.conv = SAGEConv(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.with_residual = with_residual
        self.dropout = nn.Dropout(dropout)

        self.pooling_layer = (
            SAGPooling(output_dim, ratio=pooling_ratio) if pooling_ratio else None
        )

        self.output_dim = output_dim

    def forward(self, d: Data) -> Data:
        x, edge_index, batch = d.x, d.edge_index, d.batch
        f = self.conv(x, edge_index)
        f = F.gelu(f)
        f = self.norm(f)
        f = self.dropout(f)

        if self.pooling_layer:
            x, edge_index, _, batch, _, _ = self.pooling_layer(
                f, edge_index, batch=batch
            )

        if self.with_residual:
            f += x

        new_data = Data(x=f, edge_index=edge_index, batch=batch)
        return d.update(new_data)


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
        pooling_ratio: float | None = None,
    ):
        super().__init__()

        self.gcns = nn.ModuleList()

        block = SAGEBlock(
            graph_input_dim,
            dropout=dropout,
            with_residual=False,
            output_dim=sage_channels,
        )

        for _ in range(sage_layers):
            graph_input_dim = block.output_dim
            block = SAGEBlock(
                graph_input_dim,
                dropout=dropout,
                with_residual=True,
                pooling_ratio=pooling_ratio,
                output_dim=sage_channels,
            )
            self.gcns.append(block)

        self.mlp = nn.Sequential(
            LinearBlock(
                self.gcns[-1].output_dim,
                linear_channels,
                with_residual=False,
                dropout=dropout,
            ),
            *[
                LinearBlock(linear_channels, linear_channels, dropout=dropout)
                for _ in range(linear_layers)
            ],
            nn.Linear(linear_channels, 1),
            # What if we.. softmaxed across the batch? that would force diversity,
            # but how does it make any sense?
        )

    def forward(self, data: Data) -> torch.Tensor:
        d = data
        for gcn_block in self.gcns:
            d = gcn_block(d)

        max_pool = global_max_pool(d.x, d.batch)
        return self.mlp(max_pool)
