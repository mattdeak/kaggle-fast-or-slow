import os

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from ml.xla_gcn_v1.dataset import XLATileDataset
from ml.xla_gcn_v1.model import ModifiedGCN

# |%%--%%| <I0yQRtnqYq|NhXS6Kuh4Q>

INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24

GCN_DIMS = [64, 32, 16]
LINEAR_DIMS = [128, 64, 32]

nn = ModifiedGCN(INPUT_DIM, GLOBAL_INPUT_DIM, GCN_DIMS, LINEAR_DIMS, 1)

# |%%--%%| <NhXS6Kuh4Q|1NKjfOoHTI>

TRAIN_DIR = "data/npz/tile/xla/train"
VALID_DIR = "data/npz/tile/xla/valid"
train_dataset = XLATileDataset(
    processed="data/processed/train",
    raw=TRAIN_DIR,
    max_files_per_config=200,
)

valid_dataset = XLATileDataset(processed="data/processed/valid", raw=VALID_DIR)


# |%%--%%| <1NKjfOoHTI|w6oI8NpWeo>

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4)

# |%%--%%| <w6oI8NpWeo|fQ28csLHaF>


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


LR = 0.0001
WEIGHT_DECAY = 5e-4

model = nn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


LOG_INTERVAL = 100
EVAL_INTERVAL = 20000
CHECKPOINT_INTERVAL = 20000
MODEL_DIR = "models/gcn_v1_test"
os.makedirs(MODEL_DIR, exist_ok=True)  # type: ignore


# |%%--%%| <fQ28csLHaF|HwDaQ6K4aZ>

import wandb

model.train()
with wandb.init(
    project="kaggle-fast-or-slow",
    name="gcn_v1_test_mean_max_pool",
    job_type="test",
    config={
        "model": "gcn_v1",
        "dataset": "xla",
        "optimizer": "adam",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
    },
):
    wandb.watch(model)
    model.train()
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out.flatten(), batch.y)
        loss.backward()
        optimizer.step()

        if i % LOG_INTERVAL == 0:
            wandb.log({"train_loss": loss.item()})

        if i % EVAL_INTERVAL == 0:
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
            wandb.log({"valid_loss": validation_loss})
            model.train()

        if i % CHECKPOINT_INTERVAL == 0:
            print("Saving checkpoint...")
            model_path = os.path.join(MODEL_DIR, f"model_step{i}.pt")
            optim_path = os.path.join(MODEL_DIR, f"optimizer_step{i}.pt")
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optim_path)
