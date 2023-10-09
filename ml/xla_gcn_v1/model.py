import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class ModifiedGCN(torch.nn.Module):
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

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.leaky_relu(x)

        pool = global_mean_pool(x, data.batch)
        x = torch.cat((pool, global_features), dim=1)

        for fc in self.fcs:
            x = fc(x)
            x = F.leaky_relu(x)

        x = self.output_layer(x)
        return x
