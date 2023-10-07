import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm.auto import tqdm


class ModifiedGCN(torch.nn.Module):
    def __init__(
        self,
        graph_input_dim: int,
        global_input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.conv1 = GCNConv(graph_input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 8)

        self.fc1 = torch.nn.Linear(global_input_dim + 8, 16)
        self.fc2 = torch.nn.Linear(16, output_dim)

    def forward(self, data: Data):
        x, edge_index, global_features = (
            data.x,
            data.edge_index,
            data["global_features"],
        )
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        pool = global_mean_pool(x, data.batch)
        concat = torch.cat((pool, global_features), dim=1)

        x = self.fc1(concat)

        x = F.leaky_relu(x)
        x = self.fc2(x)

        x = F.leaky_relu(x)
        return x
