import random
from typing import Generator

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm.auto import tqdm

from ml.xla_gcn_v1.dataset import XLATileDataset, parse_file

# |%%--%%| <Qm2sDuudhp|I0yQRtnqYq>


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


# |%%--%%| <I0yQRtnqYq|NhXS6Kuh4Q>

INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24

nn = ModifiedGCN(INPUT_DIM, GLOBAL_INPUT_DIM, 16, 1)

# |%%--%%| <NhXS6Kuh4Q|rB1YNoqy5j>
DIR = "data/npz/tile/xla/train"

# |%%--%%| <rB1YNoqy5j|nfOme9DOd0>

from torch_geometric.data import DataLoader

loader = DataLoader(test_data, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()

for batch in tqdm(loader):
    batch = batch.to(device)
    optimizer.zero_grad()
    out = model(batch)
    loss = F.mse_loss(out.flatten(), batch.y)
    loss.backward()
    optimizer.step()


# |%%--%%| <nfOme9DOd0|1NKjfOoHTI>


train_dataset = XLATileDataset(
    processed="data/processed", raw="data/npz/tile/xla/train"
)
