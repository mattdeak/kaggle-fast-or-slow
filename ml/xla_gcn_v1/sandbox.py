import os

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

GCN_DIMS = [32, 24, 16]
LINEAR_DIMS = [64, 32, 16]

nn = ModifiedGAT(INPUT_DIM, GLOBAL_INPUT_DIM, GCN_DIMS, LINEAR_DIMS, 1)

# |%%--%%| <NhXS6Kuh4Q|1NKjfOoHTI>

TRAIN_DIR = "data/npz/tile/xla/train"
VALID_DIR = "data/npz/tile/xla/valid"
train_dataset = XLATileDataset(
    processed="data/processed/train",
    raw=TRAIN_DIR,
    max_files_per_config=1000,
)

valid_dataset = XLATileDataset(processed="data/processed/valid", raw=VALID_DIR)


# |%%--%%| <1NKjfOoHTI|w6oI8NpWeo>

BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4)

# |%%--%%| <w6oI8NpWeo|fQ28csLHaF>


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


LR = 0.0001
WEIGHT_DECAY = 5e-4

model = nn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


LOG_INTERVAL = 100
EVAL_INTERVAL = 20000
CHECKPOINT_INTERVAL = 20000
MODEL_DIR = f"models/{nn.MODEL_ID}"
os.makedirs(MODEL_DIR, exist_ok=True)  # type: ignore

EPOCHS = 3

# |%%--%%| <fQ28csLHaF|HwDaQ6K4aZ>

import wandb

with wandb.init(
    project="kaggle-fast-or-slow",
    # id="gat_v1_test_mean_max_pool",
    name=f"{nn.MODEL_ID}_test_max_pool",
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
    notes="Simple GAT with LayerNorm and global max pooling on graph layers",
):
    wandb.watch(model)
    model.train()
    for epoch in range(EPOCHS):
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
        model_path = os.path.join(MODEL_DIR, f"model_epoch{epoch}.pt")
        optim_path = os.path.join(MODEL_DIR, f"optimizer_epoch{epoch}.pt")
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_final.pt"))
torch.save(optimizer.state_dict(), os.path.join(MODEL_DIR, "optimizer_final.pt"))
