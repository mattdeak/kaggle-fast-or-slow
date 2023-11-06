from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv
from torch_geometric.nn.pool import global_mean_pool

INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24

GlobalPoolingFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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

        self.output_dim = output_dim

    def forward(self, d: Data) -> Data:
        x, edge_index, batch = d.x, d.edge_index, d.batch

        f = self.conv(x, edge_index)
        f = F.gelu(f)
        f = self.norm(f)
        f = self.dropout(f)

        if self.with_residual:
            f += x

        new_data = Data(x=f, edge_index=edge_index, batch=batch)
        return d.update(new_data)


class GATBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        heads: int = 4,
        with_residual: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv = GATConv(input_dim, output_dim, heads=heads)
        self.norm = nn.LayerNorm(output_dim * heads)
        self.with_residual = with_residual
        self.dropout = nn.Dropout(dropout)

        self.output_dim = output_dim * heads

    def forward(self, d: Data) -> Data:
        x, edge_index, batch = d.x, d.edge_index, d.batch

        f = self.conv(x, edge_index)
        f = F.gelu(f)
        f = self.norm(f)
        f = self.dropout(f)

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
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.with_residual = with_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.linear(x)
        f = F.gelu(f)
        f = self.norm(f)
        f = self.dropout(f)

        if self.with_residual:
            f += x

        return f


class GraphMLP(nn.Module):
    def __init__(
        self,
        graph_input_dim: int = INPUT_DIM,
        global_features_dim: int | None = GLOBAL_INPUT_DIM,
        graph_channels: int = 64,
        graph_layers: int = 6,
        linear_channels: int = 32,
        linear_layers: int = 3,
        dropout: float = 0.2,
        pooling_fn: GlobalPoolingFn = global_mean_pool,
        pooling_feature_multiplier: int = 1,
        graph_conv: Literal["sage", "gat"] = "sage",
        graph_conv_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.pooling_fn = pooling_fn
        self.gcns = nn.ModuleList()

        conv = SAGEBlock if graph_conv == "sage" else GATBlock
        block = conv(
            graph_input_dim,
            dropout=dropout,
            with_residual=False,
            output_dim=graph_channels,
            **(graph_conv_kwargs or {}),
        )
        self.gcns.append(block)

        for _ in range(graph_layers):
            graph_input_dim = block.output_dim
            block = conv(
                graph_input_dim,
                output_dim=graph_channels,
                dropout=dropout,
                with_residual=False,  # doesn't play well with pooling
            )
            self.gcns.append(block)

        if global_features_dim:
            first_linear_input_dim = (
                pooling_feature_multiplier * block.output_dim + global_features_dim
            )
        else:
            first_linear_input_dim = pooling_feature_multiplier * block.output_dim

        self.mlp = nn.Sequential(
            LinearBlock(
                first_linear_input_dim,
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
        d = data
        for gcn_block in self.gcns:
            d = gcn_block(d)

        # TODO: this is specific to the pooling function we made. We should make
        # sure our poolers are interchangeable.
        pool = self.pooling_fn(d.x, d.batch)  # type: ignore

        if data.global_features is not None:
            # shape we need from global features is (batch, global_features_dim)
            # shape we have is (batch * global_features_dim)
            global_features = data.global_features.reshape(pool.shape[0], -1)
            pool = torch.cat([pool, global_features], dim=1)

        return self.mlp(pool)
