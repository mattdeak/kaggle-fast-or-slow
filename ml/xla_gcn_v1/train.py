import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

import wandb
from lib.transforms.node_sum_pooling_with_graph_features_v1 import get_data
from ml.xla_gcn_v1.dataset import XLATileDataset
from ml.xla_gcn_v1.model import ModifiedGAT

TRAIN_DIR = "data/npz/tile/xla/train"
VALID_DIR = "data/npz/tile/xla/valid"

# Defined by our feature space
INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24

LOG_INTERVAL = 100

MODEL_ROOT_DIR = "models"


def train_gat(
    learning_rate: float,
    linear_layers: list[int],
    graph_layers: list[int],
    batch_size: int = 64,
    name: str = "gat_default",
    epochs: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = XLATileDataset(
        processed="data/processed/train",
        raw=TRAIN_DIR,
        max_files_per_config=2000,
    )

    valid_dataset = XLATileDataset(processed="data/processed/valid", raw=VALID_DIR)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4)

    model = ModifiedGAT(INPUT_DIM, GLOBAL_INPUT_DIM, graph_layers, linear_layers, 1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, fused=True)

    with wandb.init(
        project="kaggle-fast-or-slow",
        name=name,
        job_type="test",
        config={
            "model": model.MODEL_ID,
            "dataset": "xla",
            "optimizer": "adam",
            "lr": learning_rate,
            "batch_size": batch_size,
            "graph_dims": graph_layers,
            "linear_dims": linear_layers,
        },
        notes="Simple GAT with LayerNorm and global mean pooling on graph layers",
    ):
        wandb.watch(model)
        model.train()
        for epoch in range(epochs):
            model.train()
            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)

                loss = F.mse_loss(out.flatten(), batch.y)
                loss.backward()
                optimizer.step()

                if i % LOG_INTERVAL == 0:
                    train_rmse = np.sqrt(loss.item())
                    wandb.log({"train_rmse": train_rmse})

            print("Evaluating...")
            model.eval()
            validation_loss = 0
            with torch.no_grad():
                for batch in valid_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    loss = F.mse_loss(out.flatten(), batch.y)
                    validation_loss += loss.item()

            validation_loss /= len(valid_loader)
            validation_loss = np.sqrt(validation_loss)
            wandb.log({"valid_rmse": validation_loss})

            print("Saving checkpoint...")
