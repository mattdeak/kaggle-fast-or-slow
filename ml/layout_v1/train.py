import os

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

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
)
dataset.load()

val_dataset = LayoutDataset(
    directories=val_directories,
)
val_dataset.load()


loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# |%%--%%| <4MlM0FfI0e|0uhA8hyj2Z>

model = SAGEMLP(graph_input_dim=GRAPH_DIM, sage_layers=1, linear_layers=1, dropout=0.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=0.01)


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
    num_samples = 0

    with torch.no_grad():
        for i, eval_batch in enumerate(loader):
            if i >= num_batches:
                break
            eval_batch = eval_batch.to(device)
            output = model(eval_batch)
            loss = criterion(output, eval_batch.y)
            total_loss += loss.item() * eval_batch.num_graphs
            num_samples += eval_batch.num_graphs

    avg_loss = total_loss / num_samples
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

    criterion = nn.MSELoss()
    heap = []
    N = 5  # Number of recent checkpoints to keep
    best_eval_loss = float("inf")

    for iter_count in range(MAX_ITERS):
        model.train()
        for batch in loader:
            batch = batch.to(device)

            # Forward Pass
            output = model(batch)

            # Compute Loss
            loss = criterion(output, batch.y)

            # Zero Gradients, Perform a Backward Pass, Update Weights
            optim.zero_grad()
            loss.backward()
            optim.step()

            if iter_count % LOG_INTERVAL == 0:
                print(f"Iteration {iter_count} | Loss: {loss.item()}")
                wandb.log({"train_loss": loss.item()})

            # Checkpointing
            if iter_count % EVAL_INTERVAL == 0:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "iteration": iter_count,
                }
                torch.save(checkpoint, f"checkpoint_{iter_count}.pth")
                wandb.save(f"checkpoint_{iter_count}.pth")

            if iter_count >= MAX_ITERS:
                break

            # Evaluation Loop
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

            if iter_count >= MAX_ITERS:
                break
