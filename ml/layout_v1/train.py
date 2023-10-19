import argparse
import heapq
import os
from dataclasses import dataclass
from typing import Any

import scipy.stats as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

import wandb
from ml.layout_v1.dataset import LayoutDataset
from ml.layout_v1.model import SAGEMLP
from ml.layout_v1.sampler import ConfigCrossoverBatchSampler

# ---- Config ---- #
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")
print("Using device:", device)

# Logging/Metrics
LOG_INTERVAL = 500
MAX_ITERS = 200000
EVAL_ITERS = 512  # per loader
EVAL_INTERVAL = 5000

# Model hyperparameters
SAGE_LAYERS = 8
SAGE_CHANNELS = 256
LINEAR_LAYERS = 4
LINEAR_CHANNELS = 256
DROPOUT = 0.1

# Optimizer
LR = 3e-4
WEIGHT_DECAY = 1e-4
MARGIN = 0.1  # penalize by 0.1


# Training Details
BATCH_SIZE = 4  # pretty low cause memory is hard
NUM_WORKERS = 4
XLA_DATA_DIR = "data/layout/xla"
NLP_DATA_DIR = "data/layout/nlp"
DATA_DIRS = [
    XLA_DATA_DIR
]  # only xla this run. I think it may be nonsense to train on both
CATEGORIES = ["default", "random"]  # I think this is fine though?

# Deterministic
GRAPH_DIM = 279

# Training Mods
USE_AMP = False  # seems broken?
PROFILE = False
WANDB_LOG = True
SAVE_CHECKPOINTS = True
DATASET_MODE = "memmapped"  # memmapped or in-memory
ATTEMPT_OVERFIT = False  # good for validating learning behaviour
OVERFIT_DATASET_SIZE = 1024

# ---- Data ---- #
directories = [
    os.path.join(data_dir, category, "train")
    for data_dir in DATA_DIRS
    for category in CATEGORIES
]

val_directories = [
    os.path.join(data_dir, category, "valid")
    for data_dir in DATA_DIRS
    for category in CATEGORIES
]


dataset = LayoutDataset(
    directories=directories, mode=DATASET_MODE, processed_dir="data/processed_layout"
)
dataset.load()

if ATTEMPT_OVERFIT:
    dataset = dataset[:OVERFIT_DATASET_SIZE]

# We break these up because the distributions are different,
# so we may want to analyze the metrics separately
default_val_xla_dataset = LayoutDataset(
    directories=[os.path.join(XLA_DATA_DIR, "default", "valid")],
    mode=DATASET_MODE,
    processed_dir="data/processed_layout",
)
random_val_xla_dataset = LayoutDataset(
    directories=[os.path.join(XLA_DATA_DIR, "random", "valid")],
    mode=DATASET_MODE,
    processed_dir="data/processed_layout",
)


default_val_xla_dataset.load()
random_val_xla_dataset.load()

if ATTEMPT_OVERFIT:
    # we need to slice the idx groups too
    train_idx_groups = sorted(dataset.idx_groups, key=lambda x: sum(x))
    # this is nested. We want to slice by a running total
    idx_groups = []
    remaining = OVERFIT_DATASET_SIZE
    for group in train_idx_groups:
        if len(group) > remaining:
            idx_groups.append(group[:remaining])
            break
        else:
            idx_groups.append(group)
            remaining -= len(group)
else:
    train_idx_groups = dataset.idx_groups

train_sampler = ConfigCrossoverBatchSampler(
    groups=train_idx_groups,
    batch_size=BATCH_SIZE,
    shuffle_groups=True,
    shuffle_within_groups=True,
)
default_val_sampler = ConfigCrossoverBatchSampler(
    groups=default_val_xla_dataset.idx_groups,
    batch_size=BATCH_SIZE,
    shuffle_groups=True,
    shuffle_within_groups=True,
)
random_val_sampler = ConfigCrossoverBatchSampler(
    groups=random_val_xla_dataset.idx_groups,
    batch_size=BATCH_SIZE,
    shuffle_groups=True,
    shuffle_within_groups=True,
)


def make_dataloader(
    dataset: LayoutDataset, sampler: ConfigCrossoverBatchSampler
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,  # this type errors but it's fine, it's cause torch geometric dataloader is dumb
        shuffle=False,
        batch_sampler=sampler,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )


loader = make_dataloader(dataset, train_sampler)

eval_loaders = {
    "default_xla": make_dataloader(default_val_xla_dataset, default_val_sampler),
    "random_xla": make_dataloader(random_val_xla_dataset, random_val_sampler),
}


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


@dataclass
class EvalMetrics:
    avg_loss: float
    avg_kendall_tau: float
    std_kendall_tau: float


@torch.no_grad()
def evaluate(
    model: nn.Module,
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
            y = eval_batch.y
            # generate pairs for margin ranking loss
            pairs = torch.combinations(torch.arange(output.shape[0]), 2).to(device)
            y_true = torch.where(y[pairs[:, 0]] > y[pairs[:, 1]], 1, -1).to(device)

            loss = F.margin_ranking_loss(
                output[pairs[:, 0]].flatten(),
                output[pairs[:, 1]].flatten(),
                y_true,
                margin=MARGIN,
            )

        total_loss += loss.item()
        num_iters += 1

    avg_loss = total_loss / num_iters
    return avg_loss


def train_batch(
    model: nn.Module,
    batch,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
) -> float:
    optim.zero_grad()

    # Forward Pass
    with torch.autocast(device_type=device, enabled=USE_AMP):
        output = model(batch)
        y = batch.y
        # Compute Loss
        pairs = torch.combinations(torch.arange(output.shape[0]), 2)
        y_true = torch.where(y[pairs[:, 0]] > y[pairs[:, 1]], 1, -1).to(device)

        loss = F.margin_ranking_loss(
            output[pairs[:, 0]].flatten(),
            output[pairs[:, 1]].flatten(),
            y_true,
            margin=MARGIN,
        )

    train_loss = loss.item()

    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    return train_loss


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
            "data_dir": DATA_DIRS,
            "categories": CATEGORIES,
            "amp": USE_AMP,
            "attempt_overfit": ATTEMPT_OVERFIT,
            "job_type": "layout",
            "subtype": "train",
        },
        mode="online" if WANDB_LOG else "disabled",
    ):
        wandb.watch(model)

        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)  # type: ignore
        avg_loss = 0

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
                checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
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
                avg_loss += train_batch(model, batch, optim, scaler)

            if iter_count % LOG_INTERVAL == 0 and iter_count > 0:
                avg_loss /= LOG_INTERVAL
                print(f"Iteration {iter_count} | Avg Loss: {avg_loss}")
                wandb.log({"train_loss": avg_loss})
                avg_loss = 0

            # Evaluation Loop and Checkpointing
            if iter_count % EVAL_INTERVAL == 0 and iter_count > 0:
                model.eval()
                with record_function("evaluate"):
                    random_xla_eval_loss = evaluate(
                        model, eval_loaders["random_xla"], EVAL_ITERS, device
                    )

                    default_xla_eval_loss = evaluate(
                        model,
                        eval_loaders["default_xla"],
                        EVAL_ITERS,
                        device,
                    )

                wandb.log(
                    {
                        "xla_random_val_loss": random_xla_eval_loss,
                        "xla_default_val_loss": default_xla_eval_loss,
                        "avg_val_loss": (random_xla_eval_loss + default_xla_eval_loss)
                        / 2,
                    }
                )

                model.train()

                if not SAVE_CHECKPOINTS:
                    continue

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "iteration": iter_count,
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
    args = parser.parse_args()
    if PROFILE:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            run(id=args.id)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        run(id=args.id)
