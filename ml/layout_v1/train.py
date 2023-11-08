import argparse
import heapq
import os
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import scipy.stats as ss
import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Batch
from tqdm.auto import tqdm

import wandb
from ml.layout_v1.checkpointer import Checkpointer
from ml.layout_v1.job.builder import RunConfig, instantiate_from_spec
from ml.layout_v1.job.spec import JobSpec, ProcessorSpec
from ml.layout_v1.utils import get_rank

# ---- Config ---- #
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")
print("Using device:", device)

DEFAULT_CONFIG = JobSpec(
    dataset_types=["xla"],
    dataset_subtypes=["default", "random"],
    processed_directory="data/processed",
    log_interval=1000,
    log_table_interval=20000,
    eval_interval=10000,
    eval_iterations=512,
    epochs=6,
    graph_layers=3,
    graph_channels=128,
    linear_layers=3,
    linear_channels=128,
    dropout=0.0,
    graph_convolution_type="gat",
    graph_convolution_kwargs={"heads": 4},
    batch_size=16,
    num_workers=4,
    use_amp=False,
    wandb=True,
    save_checkpoints=True,
    optimizer="adamw",
    optimizer_kwargs={"lr": 3e-4, "weight_decay": 0.01},
    criterion="margin-loss",
    criterion_kwargs={"margin": 1.0},
    # processors
    preprocessors=ProcessorSpec(
        graph="config-communities",
        graph_kwargs={"hops": 2},
        node="node-processor",
        config="config-feature-generator",
        opcode="group-ohe-embedder",
    ),
    postprocessors=ProcessorSpec(
        graph=None,
        node=None,
        config=None,
        opcode=None,
        global_=None,
        target=None,
    ),
    # crossover
    crossover=0.0,
)


@dataclass
class EvalMetrics:
    avg_loss: float
    avg_kendall_tau: float
    std_kendall_tau: float


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_batches: int,
    device: str,
) -> EvalMetrics:
    total_loss = 0
    num_iters = 0

    kendall_taus: list[float] = []

    for i, eval_batch in tqdm(enumerate(loader), total=min(num_batches, len(loader))):
        if i >= num_batches:
            break

        eval_batch = eval_batch.to(device)
        try:
            with torch.autocast(device_type=device, enabled=run_config.use_amp):  # type: ignore
                output = model(eval_batch)
                y = eval_batch.y
                # generate pairs for margin ranking loss
                loss = criterion(
                    output.squeeze(),
                    y.squeeze(),
                )
                # loss = listMLEalt(output.squeeze(), y.squeeze())

                predicted_rank = get_rank(output.flatten()).cpu().numpy()
                true_rank = get_rank(y.flatten()).cpu().numpy()

                kendall_tau: float = cast(
                    float,
                    ss.kendalltau(  # type: ignore
                        predicted_rank, true_rank
                    ).correlation,
                )
                kendall_taus.append(kendall_tau)
        except:
            print("Failed to evaluate batch. Batch: ", eval_batch)
            raise

        total_loss += loss.item()
        num_iters += 1

    avg_loss = total_loss / num_iters
    return EvalMetrics(
        avg_loss, float(np.mean(kendall_taus)), float(np.std(kendall_taus))
    )


def train_batch(
    model: torch.nn.Module,
    batch: Batch,
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    use_amp: bool = False,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    optim.zero_grad()

    # Forward Pass
    with torch.autocast(device_type=device, enabled=use_amp):  # type: ignore
        output = model(batch)

        y = batch.y  # type: ignore
        # generate pairs for margin ranking loss
        loss: torch.Tensor = criterion(
            output.squeeze(),
            y.squeeze(),  # type: ignore
        )
        # loss = listMLEalt(output.squeeze(), y.squeeze())

    train_loss = loss.item()

    # The type stubs suck
    scaler.scale(loss).backward()  # type: ignore
    scaler.step(optim)  # type: ignore
    scaler.update()  # type: ignore

    output = output.flatten().detach()
    y = y.detach()  # type: ignore

    return train_loss, output, y  # type: ignore


@torch.no_grad()
def log_train_metrics(
    *,
    output: torch.Tensor,
    y: torch.Tensor,
    train_loss: float,
    log_table: bool = False,
) -> None:
    cpu_output = output.cpu().flatten().numpy()
    cpu_y = y.cpu().flatten().numpy()
    ranked = get_rank(cpu_output)
    true_ranked = get_rank(cpu_y)

    data = [
        (cpu_output[i], cpu_y[i], ranked[i], true_ranked[i])
        for i in range(len(cpu_output))
    ]
    kendall_tau: float = ss.kendalltau(ranked, true_ranked).correlation  # type: ignore
    if log_table:
        wandb.log(  # type: ignore
            {
                "train_example": wandb.Table(
                    columns=["output", "y", "predicted_rank", "true_rank"],
                    data=data,
                ),
                "train/kendall_tau": kendall_tau,
                "train/loss": train_loss,
            }
        )
    else:
        wandb.log(  # type: ignore
            {"train/kendall_tau": kendall_tau, "train/loss": train_loss}
        )


@torch.no_grad()
def log_eval_metrics(
    *,
    results: dict[str, EvalMetrics],
    is_full: bool = False,
):
    """Log the evaluation metrics. Each separately and then all averages."""
    if is_full:
        prefix = "full"
    else:
        prefix = "eval"

    for eval_type, eval_metrics in results.items():
        wandb.log(  # type: ignore
            {
                f"{prefix}/{eval_type}/loss": eval_metrics.avg_loss,
                f"{prefix}/{eval_type}/kendall_tau": eval_metrics.avg_kendall_tau,
                f"{prefix}/{eval_type}/kendall_tau_std": eval_metrics.std_kendall_tau,
            }
        )

    avg_loss = np.mean([m.avg_loss for m in results.values()])
    avg_kendall_tau = np.mean([m.avg_kendall_tau for m in results.values()])
    avg_kendall_tau_std = np.mean([m.std_kendall_tau for m in results.values()])

    wandb.log(  # type: ignore
        {
            f"{prefix}/avg/loss": avg_loss,
            f"{prefix}/avg/kendall_tau": avg_kendall_tau,
            f"{prefix}/avg/kendall_tau_std": avg_kendall_tau_std,
        }
    )


def run_full_epoch(
    *,
    model: torch.nn.Module,
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    train_loader: DataLoader,
    eval_loaders: dict[str, DataLoader],
    run_config: RunConfig,
    epoch: int,
    checkpointer: Checkpointer,
    start_iter: int = 0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    print(f"Starting: {epoch}")
    model.train()
    train_loader.batch_sampler.reset()  # type: ignore

    avg_loss = 0

    for iter_count, batch in tqdm(
        enumerate(train_loader, start=start_iter),
        total=len(train_loader),
    ):
        batch = batch.to(device)

        batch_loss, output, y = train_batch(
            model,
            batch,
            criterion,
            optim,
            scaler,
        )

        if scheduler is not None:
            scheduler.step()

        # scheduler.step()
        avg_loss += batch_loss

        if iter_count % run_config.log_interval == 0 and iter_count > 0:
            avg_loss /= run_config.log_interval
            print(f"Epoch {epoch+1} | Iteration {iter_count} | Avg Loss: {avg_loss}")
            # also record the most recent outputs for examination
            log_train_metrics(
                output=output,
                y=y,
                train_loss=avg_loss,
                log_table=iter_count % run_config.log_table_interval == 0,
            )
            avg_loss = 0

        # Evaluation Loop and Checkpointing
        if iter_count % run_config.eval_interval == 0 and iter_count > 0:
            model.eval()
            metrics = {}
            for eval_type, eval_loader in eval_loaders.items():
                metrics[eval_type] = evaluate(
                    model,
                    eval_loader,
                    criterion,
                    run_config.eval_iterations,
                    device,
                )
            log_eval_metrics(results=metrics)
            model.train()

        if run_config.save_checkpoints:
            checkpointer.save_checkpoint(iteration=iter_count, epoch=epoch)


def run(config: dict[str, Any] | JobSpec = DEFAULT_CONFIG, id: str | None = None):
    if isinstance(config, dict):
        config = JobSpec(**config)

    with wandb.init(
        project="kaggle-fast-or-slow",
        id=id,
        config=config.dict(),
        mode="online" if config.wandb else "disabled",
        resume="allow",
    ) as run:
        assert run is not None, "Wandb run is None"

        run_data = instantiate_from_spec(config)

        model = run_data.model
        optim = run_data.optimizer
        scheduler = run_data.scheduler
        criterion = run_data.criterion

        train_loader = run_data.train_loader
        eval_loaders = run_data.eval_loaders

        run_config = run_data.run_config

        del run_data

        model.to(device)
        wandb.watch(model)  # type: ignore

        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)  # type: ignore
        checkpointer = Checkpointer(
            checkpoint_dir=f"models/{run.id}",  # type: ignore
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            max_checkpoints=run_config.max_checkpoints,
        )

        latest = checkpointer.get_most_recent_checkpoint()

        if latest is not None:
            checkpointer.load_checkpoint(latest)
            start_iter = latest["iteration"]
            start_epoch = latest["epoch"]
        else:
            start_iter, start_epoch = 0, 0

        model.train()
        for epoch in range(start_epoch, run_config.epochs):
            run_full_epoch(
                model=model,
                criterion=criterion,
                optim=optim,
                scaler=scaler,
                train_loader=train_loader,
                eval_loaders=eval_loaders,
                checkpointer=checkpointer,
                run_config=run_config,
                epoch=epoch,
                start_iter=start_iter,
                scheduler=scheduler,
            )

            # Run full validation at the end of each epoch
            model.eval()
            metrics = {}
            for eval_type, eval_loader in eval_loaders.items():
                metrics[eval_type] = evaluate(
                    model,
                    eval_loader,
                    criterion,
                    len(eval_loader),
                    device,
                )
            log_eval_metrics(results=metrics, is_full=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None)
    args = parser.parse_args()
    run(id=args.id)
