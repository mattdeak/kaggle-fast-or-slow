import argparse
import os
from collections.abc import Iterable
from typing import Any, Generic, TypeVar, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import SAGEConv
from tqdm.auto import tqdm

import wandb
from ml.xla_gcn_v1.dataset import XLATileDataset

# |%%--%%| <I0yQRtnqYq|NhXS6Kuh4Q>

# Data-defined
INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Logging/Metrics
LOG_INTERVAL = 500
MAX_ITERS = 100000
EVAL_ITERS = 400
EVAL_INTERVAL = 2000

# Model hyperparameters
SAGE_LAYERS = 8
SAGE_CHANNELS = 256
LINEAR_LAYERS = 3
LINEAR_CHANNELS = 256
DROPOUT = 0.2

LR = 4e-3
WEIGHT_DECAY = 1e-4

# |%%--%%| <NhXS6Kuh4Q|1NKjfOoHTI>

TRAIN_DIR = "data/npz/tile/xla/train"
VALID_DIR = "data/npz/tile/xla/valid"
train_dataset = XLATileDataset(
    processed="data/processed/train",
    raw=TRAIN_DIR,
    max_files_per_config=2000,
)

valid_dataset = XLATileDataset(processed="data/processed/valid", raw=VALID_DIR)

# |%%--%%| <1NKjfOoHTI|w6oI8NpWeo>

BATCH_SIZE = 256

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
)


def cycle(iterable: Iterable[Any]):
    while True:
        for x in iterable:
            yield x


train_cycler = cycle(train_loader)

# |%%--%%| <w6oI8NpWeo|fQ28csLHaF>


# |%%--%%| <fQ28csLHaF|cQKG7nu2vX>


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
        # self.convsage = SAGEConv(input_dim, output_dim)
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
        global_input_dim: int = GLOBAL_INPUT_DIM,
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
                sage_channels + global_input_dim,
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
        global_features: torch.Tensor = data["global_features"]

        for gcn_block in self.gcns:
            x = gcn_block(x, edge_index)

        sum_pool = global_add_pool(x, batch)
        concat = torch.cat((sum_pool, global_features), dim=1)
        return self.mlp(concat)


# |%%--%%| <cQKG7nu2vX|olMpzXENxJ>


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAGEMLP(
    sage_layers=SAGE_LAYERS,
    sage_channels=SAGE_CHANNELS,
    linear_layers=LINEAR_LAYERS,
    linear_channels=LINEAR_CHANNELS,
    dropout=DROPOUT,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# |%%--%%| <olMpzXENxJ|yrLVjmoI70>

T = TypeVar("T", int, float)


class Accumulator(Generic[T]):
    def __init__(self):
        self._values: list[T] = []

    def add(self, value: T):
        self._values.append(value)

    def mean(self) -> float:
        return np.mean(self._values)

    def reset(self):
        self._values = []


def train_and_eval(save_dir: str | None = None):
    train_cycler = cycle(train_loader)
    acc = Accumulator()
    model.train()
    for i, batch in tqdm(enumerate(train_cycler), total=MAX_ITERS):
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
            wandb.log({"train_rmse": train_rmse})
            print(f"Train RMSE: {train_rmse}")

        if i % EVAL_INTERVAL == 0:
            print("Evaluating...")
            model.eval()
            validation_loss = 0
            num_eval = 0
            with torch.no_grad():
                for j, batch in enumerate(valid_loader):
                    if j > EVAL_ITERS:
                        break

                    num_eval += 1
                    batch = batch.to(device)
                    out = model(batch)
                    loss = F.mse_loss(out.flatten(), batch.y)
                    validation_loss += loss.item()

            validation_loss /= num_eval
            validation_loss = np.sqrt(validation_loss)
            wandb.log({"validation_rmse": validation_loss})

            if save_dir is not None:
                save_path = f"{save_dir}/{i}.pt"
                torch.save(model.state_dict(), save_path)
            print(f"Validation RMSE: {validation_loss}")
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    with wandb.init(
        project="kaggle-fast-or-slow",
        # id="gat_v1_test_mean_max_pool",
        name="sage_v1_test",
        job_type="test",
        config={
            "model": "sage",
            "dataset": "xla",
            "optimizer": "adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "sage_layers": SAGE_LAYERS,
            "sage_channels": SAGE_CHANNELS,
            "linear_layers": LINEAR_LAYERS,
            "linear_channels": LINEAR_CHANNELS,
        },
        mode="disabled" if args.disable_wandb else "online",
    ) as run:
        id = run.id
        print(f"Logging to {id}")
        save_path = f"models/{id}"
        os.makedirs(save_path, exist_ok=True)
        wandb.watch(model)
        train_and_eval(save_dir=save_path)
