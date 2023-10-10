import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GCNConv, LayerNorm, global_mean_pool
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


class ModifiedGAT(torch.nn.Module):
    MODEL_ID = "gat"

    def __init__(
        self,
        graph_input_dim: int,
        global_input_dim: int,
        gcn_out_dims: list[int],
        linear_dims: list[int],
        output_dim: int,
    ):
        super().__init__()

        initial_conv = GATv2Conv(graph_input_dim, gcn_out_dims[0])
        self.convs = torch.nn.ModuleList(
            [initial_conv]
            + [GATv2Conv(i, o) for i, o in zip(gcn_out_dims[:-1], gcn_out_dims[1:])]
        )

        self.layer_norms = torch.nn.ModuleList([LayerNorm(i) for i in gcn_out_dims])
        first_linear = torch.nn.Linear(
            gcn_out_dims[-1] + global_input_dim, linear_dims[0]
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

        for norm, conv in zip(self.layer_norms, self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x)

        # mean_pool = global_mean_pool(x, data.batch)
        max_pool = global_max_pool(x, data.batch)
        x = torch.cat((max_pool, global_features), dim=1)

        for fc in self.fcs:
            x = fc(x)
            x = F.relu(x)

        x = self.output_layer(x)
        return x
