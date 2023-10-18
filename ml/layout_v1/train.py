import argparse
import heapq
import os
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

import wandb
from ml.layout_v1.dataset import LayoutDataset
from ml.layout_v1.model import SAGEMLP

# ---- Config ---- #
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")
print("Using device:", device)

# Logging/Metrics
LOG_INTERVAL = 500
MAX_ITERS = 100000
EVAL_ITERS = 200
EVAL_INTERVAL = 2000

# Model hyperparameters
SAGE_LAYERS = 6
SAGE_CHANNELS = 256
LINEAR_LAYERS = 3
LINEAR_CHANNELS = 256
DROPOUT = 0.0

# Optimizer
LR = 3e-4
WEIGHT_DECAY = 1e-4

# Training Details
BATCH_SIZE = 16
NUM_WORKERS = 4
DATA_DIR = "data/layout/xla"
CATEGORIES = ["default", "random"]

# Deterministic
GRAPH_DIM = 279

# Training Mods
USE_AMP = False  # seems broken?
PROFILE = False
WANDB_LOG = True
SAVE_CHECKPOINTS = True

# ---- Data ---- #
directories = [os.path.join(DATA_DIR, category, "train") for category in CATEGORIES]
val_directories = [os.path.join(DATA_DIR, category, "valid") for category in CATEGORIES]


dataset = LayoutDataset(
    directories=directories, mode="memmapped", processed_dir="data/processed_layout"
)
dataset.load()

# We break these up because the distributions are different,
# so we may want to analyze the metrics separately
default_val_dataset = LayoutDataset(
    directories=[os.path.join(DATA_DIR, "default", "valid")],
    mode="memmapped",
    processed_dir="data/processed_layout",
)
random_val_dataset = LayoutDataset(
    directories=[os.path.join(DATA_DIR, "random", "valid")],
    mode="memmapped",
    processed_dir="data/processed_layout",
)
default_val_dataset.load()
random_val_dataset.load()


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)
default_val_loader = DataLoader(
    default_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)
random_val_loader = DataLoader(
    random_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


loader = cycle(loader)

# |%%--%%| <4MlM0FfI0e|0uhA8hyj2Z>

model = SAGEMLP(
    graph_input_dim=GRAPH_DIM,
    sage_layers=SAGE_LAYERS,
    sage_channels=SAGE_CHANNELS,
    linear_channels=LINEAR_CHANNELS,
    linear_layers=LINEAR_LAYERS,
    dropout=DROPOUT,
)

model = model.to(device)
model = torch.compile(model)
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# Initialize Weights and Biases
@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    num_batches: int,
    device: str,
):
    total_loss = 0
    num_iters = 0

    for i, eval_batch in tqdm(enumerate(loader), total=min(num_batches, len(loader))):
        if i >= num_batches:
            break
        eval_batch = eval_batch.to(device)

        with torch.autocast(device_type=device, enabled=USE_AMP):
            output = model(eval_batch)
            loss = criterion(output.flatten(), eval_batch.y)

        total_loss += loss.item()
        num_iters += 1

    avg_loss = total_loss / num_iters
    return avg_loss


def train_batch(
    model: nn.Module,
    batch,
    optim: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
) -> float:
    optim.zero_grad()

    # Forward Pass
    with torch.autocast(device_type=device, enabled=USE_AMP):
        output = model(batch)
        # Compute Loss
        loss = criterion(output.flatten(), batch.y)

    train_loss = loss.item()

    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    return train_loss


class Checkpointer:
    def __init__(self, checkpoint_dir: str, max_latest: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.heap: list[tuple[int, str]] = []

    def get_latest(self) -> str | None:
        checkpoints = os.listdir(self.checkpoint_dir)
        if not checkpoints:
            return None
        # Extract latest
        sorted_checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        most_recent_checkpoint = sorted_checkpoints[-1]
        return most_recent_checkpoint

    def load_dir(self) -> None:
        checkpoints = os.listdir(self.checkpoint_dir)
        if not checkpoints:
            return
        # Extract latest
        sorted_checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        most_recent_checkpoint = sorted_checkpoints[-1]

        checkpoint = torch.load(
            os.path.join(self.checkpoint_dir, most_recent_checkpoint)
        )

        return checkpoint


def run(id: str | None = None):
    with wandb.init(
        project="kaggle-fast-or-slow",
        id=id,
        config={
            "model": "SAGEMLP",
            "resume": "allow",
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
            "amp": USE_AMP,
            "job_type": "layout",
            "subtype": "dev",
        },
        mode="online" if WANDB_LOG else "disabled",
    ):
        wandb.watch(model)

        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)  # type: ignore
        avg_loss = 0

        criterion = nn.MSELoss()
        heap: list[tuple[int, str]] = []
        N = 5  # Number of recent checkpoints to keep
        start_iter = 0

        checkpoint_dir = f"models/{wandb.run.id}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load Checkpoint
        checkpoints = os.listdir(checkpoint_dir)
        # Extract latest
        if checkpoints:
            sorted_checkpoints = sorted(
                checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0])
            )

            most_recent_checkpoint = sorted_checkpoints[-1]
            print("Loading checkpoint:", most_recent_checkpoint)

            checkpoint = torch.load(
                os.path.join(checkpoint_dir, most_recent_checkpoint)
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optim.load_state_dict(checkpoint["optimizer_state_dict"])

            start_iter = checkpoint["iteration"]
            print("Resuming from iteration:", start_iter)
        else:
            print("No Checkpoint Found in:", checkpoint_dir)

            # TODO: this is technically wrong, but it's fine for now

        model.train()
        for iter_count, batch in tqdm(
            enumerate(loader, start=start_iter),
            total=MAX_ITERS - start_iter,
        ):
            batch = batch.to(device)
            if iter_count > MAX_ITERS:
                break

            # Zero Gradients, Perform a Backward Pass, Update Weights
            with record_function("train_batch"):
                avg_loss += train_batch(model, batch, optim, criterion, scaler)

            if iter_count % LOG_INTERVAL == 0:
                avg_loss /= LOG_INTERVAL
                print(f"Iteration {iter_count} | Avg Loss: {avg_loss}")
                wandb.log({"train_loss": avg_loss})
                avg_loss = 0

            # Evaluation Loop and Checkpointing
            if iter_count % EVAL_INTERVAL == 0 and iter_count > 0:
                model.eval()
                with record_function("evaluate"):
                    random_avg_eval_loss = evaluate(
                        model, criterion, random_val_loader, EVAL_ITERS, device
                    )

                    default_avg_eval_loss = evaluate(
                        model, criterion, default_val_loader, EVAL_ITERS, device
                    )
                avg_eval_loss = (random_avg_eval_loss + default_avg_eval_loss) / 2

                wandb.log(
                    {
                        "random_val_loss": random_avg_eval_loss,
                        "default_val_loss": default_avg_eval_loss,
                        "val_loss": avg_eval_loss,
                    }
                )

                model.train()

                if not SAVE_CHECKPOINTS:
                    continue

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "iteration": iter_count,
                    "eval_loss": avg_eval_loss,
                }

                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, f"checkpoint_{iter_count}.pth"),
                )

                # Manage heap for N most recent checkpoints
                if len(heap) < N:
                    heapq.heappush(heap, (-iter_count, f"checkpoint_{iter_count}.pth"))
                else:
                    _, oldest_checkpoint = heapq.heappop(heap)
                    os.remove(os.path.join(checkpoint_dir, oldest_checkpoint))
                    heapq.heappush(heap, (-iter_count, f"checkpoint_{iter_count}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None)
    if PROFILE:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            run()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        run()
