import os
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv.rgcn_conv import torch_geometric
from tqdm.auto import tqdm

from ml.xla_gcn_v1.dataset import XLATileDataset
from ml.xla_gcn_v1.model import ModifiedGAT, ModifiedGCN

# |%%--%%| <I0yQRtnqYq|NhXS6Kuh4Q>

INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24

GCN_DIMS = [16, 16, 16, 16]
GLOBAL_MLP_SIZE = 32
GLOBAL_MLP_LAYERS = 3
LINEAR_SIZE = 32
LINEAR_LAYERS = 3
DEVICE = "cuda"


# |%%--%%| <NhXS6Kuh4Q|1NKjfOoHTI>

TRAIN_DIR = "data/npz/tile/xla/train"
VALID_DIR = "data/npz/tile/xla/valid"
train_dataset = XLATileDataset(
    processed="data/processed/train", raw=TRAIN_DIR, max_files_per_config=100, limit=100
)

valid_dataset = XLATileDataset(
    processed="data/processed/valid", raw=VALID_DIR, limit=100
)


# |%%--%%| <1NKjfOoHTI|w6oI8NpWeo>

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4)


def cycle(iterable: Iterable[Any]):
    while True:
        for x in iterable:
            yield x


train_cycler = cycle(train_loader)

# |%%--%%| <w6oI8NpWeo|fQ28csLHaF>


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# |%%--%%| <fQ28csLHaF|cQKG7nu2vX>

from torch_geometric.nn import GlobalAttention, global_add_pool
from torch_geometric.nn.conv import GCNConv, SAGEConv


class SimpleMLP(nn.Module):
    def __init__(
        self, graph_input_dim: int = INPUT_DIM, global_input_dim: int = GLOBAL_INPUT_DIM
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(graph_input_dim + global_input_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x: torch.Tensor = data.x
        batch: torch.Tensor = data.batch
        global_features: torch.Tensor = data["global_features"]

        sum_pool = global_add_pool(x, batch)
        concat = torch.cat((sum_pool, global_features), dim=1)
        return self.mlp(concat)


class SAGEBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, with_residual: bool = True):
        super().__init__()
        self.conv = SAGEConv(input_dim, output_dim)
        # self.convsage = SAGEConv(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.with_residual = with_residual

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        f = self.conv(x, edge_index)
        f = cast(torch.Tensor, F.gelu(f))
        f = self.norm(f)

        if self.with_residual:
            f += x
        return f


class LinearBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, with_residual: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.with_residual = with_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.linear(x)
        f = cast(torch.Tensor, F.gelu(f))
        f = self.norm(f)

        if self.with_residual:
            f += x
        return f


class SimpleMLPwGCN(nn.Module):
    def __init__(
        self, graph_input_dim: int = INPUT_DIM, global_input_dim: int = GLOBAL_INPUT_DIM
    ):
        super().__init__()

        self.gcns = nn.ModuleList(
            [
                SAGEBlock(graph_input_dim, 64, with_residual=False),
                SAGEBlock(64, 64),
                SAGEBlock(64, 64),
                SAGEBlock(64, 64),
                SAGEBlock(64, 64),
                SAGEBlock(64, 64),
            ]
        )

        self.pooler = GlobalAttention()

        self.mlp = nn.Sequential(
            LinearBlock(64 + global_input_dim, 32, with_residual=False),
            LinearBlock(32, 32),
            nn.Linear(32, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x: torch.Tensor = data.x
        batch: torch.Tensor = data.batch
        edge_index: torch.Tensor = data.edge_index
        global_features: torch.Tensor = data["global_features"]

        for gcn_block in self.gcns:
            x = gcn_block(x, edge_index)

        sum_pool = global_add_pool(x, batch)
        concat = torch.cat((sum_pool, global_features), dim=1)
        return self.mlp(concat)


# |%%--%%| <cQKG7nu2vX|olMpzXENxJ>


LOG_INTERVAL = 100
MAX_ITERS = 5000
EVAL_ITERS = 200
EVAL_INTERVAL = 500
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleMLPwGCN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# |%%--%%| <olMpzXENxJ|yrLVjmoI70>


class Accumulator:
    def __init__(self):
        self._values = []

    def add(self, value):
        self._values.append(value)

    def mean(self):
        return np.mean(self._values)

    def reset(self):
        self._values = []


train_cycler = cycle(train_loader)
acc = Accumulator()

model.train()
for i in tqdm(range(MAX_ITERS)):
    batch = next(train_cycler)
    if i > MAX_ITERS:
        break

    batch = batch.to(device)
    optimizer.zero_grad()
    out = model(batch)

    loss = F.mse_loss(out.flatten(), batch.y)
    acc.add(loss.item())

    loss.backward()
    optimizer.step()

    if i % LOG_INTERVAL == 0:
        train_rmse = np.sqrt(acc.mean())
        acc.reset()
        # wandb.log({"train_rmse": train_rmse})
        print(f"Train RMSE: {train_rmse}")

    if i % EVAL_INTERVAL == 0:
        print("Evaluating...")
        model.eval()
        validation_loss = 0
        num_eval = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                if i > EVAL_ITERS:
                    break

                num_eval += 1
                batch = batch.to(device)
                out = model(batch)
                loss = F.mse_loss(out.flatten(), batch.y)
                validation_loss += loss.item()

        validation_loss /= num_eval
        validation_loss = np.sqrt(validation_loss)
        print(f"Validation RMSE: {validation_loss}")
        model.train()
