import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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


LR = 0.001
WEIGHT_DECAY = 0

nn = ModifiedGAT(
    INPUT_DIM,
    GLOBAL_INPUT_DIM,
    GCN_DIMS,
    GLOBAL_MLP_SIZE,
    GLOBAL_MLP_LAYERS,
    LINEAR_SIZE,
    LINEAR_LAYERS,
    1,
)
model = nn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


MODEL_DIR = f"models/{nn.MODEL_ID}"
LOG_INTERVAL = 100
MAX_ITERS = 5000
EVAL_ITERS = 200
EVAL_INTERVAL = 500
os.makedirs(MODEL_DIR, exist_ok=True)  # type: ignore


# |%%--%%| <fQ28csLHaF|HwDaQ6K4aZ>

# load latest checkpoint


import wandb

with wandb.init(
    project="kaggle-fast-or-slow",
    # id="gat_v1_test_mean_max_pool",
    name=f"{nn.MODEL_ID}_test_mean_pool",
    job_type="test",
    config={
        "model": nn.MODEL_ID,
        "dataset": "xla",
        "optimizer": "adam",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "gcnn_dims": GCN_DIMS,
        "linear_dims": LINEAR_DIMS,
    },
    notes="Simple GAT with LayerNorm and global mean pooling on graph layers",
    mode="disabled",
):
    wandb.watch(model)
    model.train()
    for i in tqdm(range(MAX_ITERS)):
        batch = next(train_cycler)
        if i > MAX_ITERS:
            break

        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)

        loss = F.mse_loss(out.flatten(), batch.y)
        loss.backward()
        optimizer.step()

        if i % LOG_INTERVAL == 0:
            train_rmse = np.sqrt(loss.item())
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
        # wandb.log({"val_rmse": validation_loss})

    # print("Saving checkpoint...")
    # model_path = os.path.join(MODEL_DIR, f"model_epoch{epoch}.pt")
    # optim_path = os.path.join(MODEL_DIR, f"optimizer_epoch{epoch}.pt")
    # torch.save(model.state_dict(), model_path)
    # torch.save(optimizer.state_dict(), optim_path)

# torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_final.pt"))
# torch.save(optimizer.state_dict(), os.path.join(MODEL_DIR, "optimizer_final.pt"))
