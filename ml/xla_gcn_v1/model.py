import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (GATv2Conv, GCNConv, LayerNorm, global_add_pool,
                                global_mean_pool)
from torch_geometric.nn.pool import global_max_pool


class ModifiedGCN(torch.nn.Module):
    MODEL_ID = "gcn_v1"

    def __init__(
        self,
        graph_input_dim: int,
        global_input_dim: int,
        gcn_out_dims: list[int],
        linear_dims: list[int],
        output_dim: int,
    ):
        super().__init__()

        initial_conv = GCNConv(graph_input_dim, gcn_out_dims[0])
        self.convs = torch.nn.ModuleList(
            [initial_conv]
            + [GCNConv(i, o) for i, o in zip(gcn_out_dims[:-1], gcn_out_dims[1:])]
        )

        self.norms = torch.nn.ModuleList([LayerNorm(i) for i in gcn_out_dims])

        first_linear = torch.nn.Linear(
            gcn_out_dims[-1] * 2 + global_input_dim, linear_dims[0]
        )

        self.fcs = torch.nn.ModuleList(
            [first_linear]
            + [torch.nn.Linear(i, o) for i, o in zip(linear_dims[:-1], linear_dims[1:])]
        )

        self.output_layer = torch.nn.Linear(linear_dims[-1], output_dim)

    def forward(self, data: Data):
        x, edge_index, global_features = (
            data.x,
            data.edge_index,
            data["global_features"],
        )

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.leaky_relu(x)

        mean_pool = global_mean_pool(x, data.batch)
        max_pool = global_max_pool(x, data.batch)

        x = torch.cat((mean_pool, max_pool, global_features), dim=1)

        for fc in self.fcs:
            x = fc(x)
            x = F.leaky_relu(x)

        x = self.output_layer(x)
        return x


class GatConvBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = GATv2Conv(input_channels, output_channels)
        self.norm = LayerNorm(output_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x)
        x = self.norm(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, layers: int):
        super().__init__()

        self.project = torch.nn.Linear(input_dim, hidden_units)
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_units, hidden_units) for _ in range(layers)]
        )
        self.norms = torch.nn.ModuleList(
            [LayerNorm(hidden_units) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        x = self.project(x)
        for norm, fc in zip(self.norms, self.fcs):
            x = fc(x)
            x = F.leaky_relu(x)
            x = norm(x)

        return x


class ModifiedGAT(torch.nn.Module):
    MODEL_ID = "gat"

    def __init__(
        self,
        graph_input_dim: int,
        global_input_dim: int,
        gcn_out_dims: list[int],
        global_mlp_hidden_size: int,
        global_mlp_layers: int,
        linear_hidden_size: int,
        linear_layers: int,
        output_dim: int,
    ):
        super().__init__()

        initial_conv = GatConvBlock(graph_input_dim, gcn_out_dims[0])
        self.convs = torch.nn.ModuleList(
            [initial_conv]
            + [GatConvBlock(i, o) for i, o in zip(gcn_out_dims[:-1], gcn_out_dims[1:])]
        )

        self.global_input_mlp = MLP(
            global_input_dim,
            global_mlp_hidden_size,
            global_mlp_layers,
        )

        self.project_global = torch.nn.Linear(global_mlp_hidden_size, gcn_out_dims[-1])

        first_linear = torch.nn.Linear(gcn_out_dims[-1], linear_hidden_size)
        self.fcs = torch.nn.ModuleList(
            [first_linear]
            + [
                torch.nn.Linear(linear_hidden_size, linear_hidden_size)
                for _ in range(1, linear_layers)
            ]
        )

        self.output_layer = torch.nn.Linear(linear_hidden_size, output_dim)

    def forward(self, data: Data):
        x, edge_index, global_features = (
            data.x,
            data.edge_index,
            data["global_features"],
        )

        for conv in self.convs:
            x = conv(x, edge_index)

        add_pool = global_add_pool(x, data.batch)
        x_global = self.global_input_mlp(global_features)
        project_global = self.project_global(x_global)

        a = add_pool + project_global

        for fc in self.fcs:
            a = fc(a)
            a = F.leaky_relu(a)

        a = self.output_layer(a)
        return a
