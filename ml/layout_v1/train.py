import heapq
import os

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

import wandb
from ml.layout_v1.dataset import LayoutDataset
from ml.layout_v1.model import SAGEMLP

# ---- Config ---- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Logging/Metrics
LOG_INTERVAL = 500
MAX_ITERS = 100000
EVAL_ITERS = 400
EVAL_INTERVAL = 2000

# Model hyperparameters
# SAGE_LAYERS = 8
# SAGE_CHANNELS = 256
# LINEAR_LAYERS = 3
# LINEAR_CHANNELS = 256
# DROPOUT = 0.2

SAGE_LAYERS = 2
SAGE_CHANNELS = 24
LINEAR_LAYERS = 1
LINEAR_CHANNELS = 24
DROPOUT = 0.2

# Optimizer
LR = 4e-3
WEIGHT_DECAY = 1e-4

# Training Details
BATCH_SIZE = 8
NUM_WORKERS = 8
DATA_DIR = "data/layout/xla"
CATEGORIES = ["default", "random"]

# Deterministic
GRAPH_DIM = 279

# ---- Data ---- #
directories = [os.path.join(DATA_DIR, category, "train") for category in CATEGORIES]
val_directories = [os.path.join(DATA_DIR, category, "valid") for category in CATEGORIES]


dataset = LayoutDataset(
    directories=directories,
    mode="lazy",
)
dataset.load()

val_dataset = LayoutDataset(
    directories=val_directories,
    mode="lazy",
)
val_dataset.load()


loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)

# |%%--%%| <4MlM0FfI0e|0uhA8hyj2Z>

model = SAGEMLP(
    graph_input_dim=GRAPH_DIM,
    sage_layers=SAGE_LAYERS,
    sage_channels=SAGE_CHANNELS,
    linear_channels=LINEAR_CHANNELS,
    linear_layers=LINEAR_LAYERS,
    dropout=DROPOUT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# Initialize Weights and Biases
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    num_batches: int,
    device: torch.device,
):
    model.eval()
    total_loss = 0
    num_iters = 0

    with torch.no_grad():
        for i, eval_batch in tqdm(
            enumerate(loader), total=min(num_batches, len(loader))
        ):
            if i >= num_batches:
                break
            eval_batch = eval_batch.to(device)
            output = model(eval_batch)
            loss = criterion(output.flatten(), eval_batch.y)
            total_loss += loss.item()
            num_iters += 1

    avg_loss = total_loss / num_iters
    return avg_loss


with wandb.init(
    project="kaggle-fast-or-slow",
    config={
        "model": "SAGEMLP",
        "sage_layers": SAGE_LAYERS,
        "sage_channels": SAGE_CHANNELS,
        "linear_layers": LINEAR_LAYERS,
        "linear_channels": LINEAR_CHANNELS,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "data_dir": DATA_DIR,
        "categories": CATEGORIES,
        "job_type": "layout",
    },
):
    wandb.watch(model)

    avg_loss = 0

    criterion = nn.MSELoss()
    heap: list[tuple[int, str]] = []
    N = 5  # Number of recent checkpoints to keep
    best_eval_loss = float("inf")

    model.train()
    for iter_count, batch in tqdm(enumerate(loader), total=min(MAX_ITERS, len(loader))):
        if iter_count > MAX_ITERS:
            break

        batch = batch.to(device)

        # Forward Pass
        output = model(batch)

        # Compute Loss
        loss = criterion(output.flatten(), batch.y)
        avg_loss += loss.item()

        # Zero Gradients, Perform a Backward Pass, Update Weights
        optim.zero_grad()
        loss.backward()
        optim.step()

        if iter_count % LOG_INTERVAL == 0:
            avg_loss /= LOG_INTERVAL
            print(f"Iteration {iter_count} | Avg Loss: {avg_loss}")
            wandb.log({"train_loss": avg_loss})
            avg_loss = 0

        # Evaluation Loop and Checkpointing
        if iter_count % EVAL_INTERVAL == 0:
            avg_eval_loss = evaluate(model, criterion, loader, EVAL_ITERS, device)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "iteration": iter_count,
                "eval_loss": avg_eval_loss,
            }
            torch.save(checkpoint, f"checkpoint_{iter_count}.pth")

            # Save best-performing checkpoint
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                torch.save(checkpoint, "best_checkpoint.pth")

            # Manage heap for N most recent checkpoints
            if len(heap) < N:
                heapq.heappush(heap, (-iter_count, f"checkpoint_{iter_count}.pth"))
            else:
                _, oldest_checkpoint = heapq.heappop(heap)
                os.remove(oldest_checkpoint)
                heapq.heappush(heap, (-iter_count, f"checkpoint_{iter_count}.pth"))

            wandb.save(f"checkpoint_{iter_count}.pth")
