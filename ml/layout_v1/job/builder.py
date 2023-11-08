import os
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from torch_geometric import torch_geometric
from torch_geometric.loader import DataLoader

from ml.layout_v1.dataset import (ConcatenatedDataset, LayoutDataset,
                                  LayoutTransforms)
from ml.layout_v1.job.constants import (CONFIG_PROCESSORS, CRITERIONS,
                                        GLOBAL_POOLINGS, GLOBAL_PROCESSORS,
                                        GRAPH_PROCESSORS, NODE_PROCESSORS,
                                        OPCODE_PROCESSORS, OPTIMIZERS,
                                        SCHEDULERS, TARGET_PROCESSORS,
                                        DatasetSubtype, DatasetType)
from ml.layout_v1.job.spec import JobSpec, ProcessorSpec
from ml.layout_v1.model import GraphMLP
from ml.layout_v1.sampler import ConfigCrossoverBatchSampler


@dataclass
class RunConfig:
    log_interval: int = 1000
    log_table_interval: int = 20000
    eval_interval: int = 10000
    eval_iterations: int = 512
    epochs: int = 6
    save_checkpoints: bool = True
    max_checkpoints: int = 5


@dataclass
class RunData:
    train_loader: DataLoader
    eval_loaders: dict[str, DataLoader]

    model: GraphMLP
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None

    run_config: RunConfig


def generate_from_config(config: dict[str, Any]) -> RunData:
    return instantiate_from_spec(JobSpec(**config))


def instantiate_from_spec(spec: JobSpec) -> RunData:
    """Generate the actual job config for a specific run."""

    preprocessors = build_processors(spec.preprocessors)
    postprocessors = build_processors(spec.postprocessors)

    train_data_directories = generate_dataset_dirs(
        dataset_types=spec.dataset_types,
        dataset_subtypes=spec.dataset_subtypes,
        split="train",
    )

    train_datasets = [
        build_dataset(d, spec.processed_directory, preprocessors, postprocessors)
        for d in train_data_directories
    ]
    train_dataset = ConcatenatedDataset(train_datasets)
    train_sampler = ConfigCrossoverBatchSampler(
        groups=train_dataset.idx_groups,
        batch_size=spec.batch_size,
        shuffle_groups=True,
        shuffle_within_groups=True,
        out_of_config_crossover_prob=spec.crossover,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # needs to be 1 for custom sampler
        shuffle=False,
        batch_sampler=train_sampler,
        pin_memory=True,
        num_workers=spec.num_workers,
    )

    eval_loaders = {}
    for ds_type in spec.dataset_types:
        for ds_subtype in spec.dataset_subtypes:
            val_dir = get_dataset_dir(ds_type, ds_subtype, split="val")
            ds = build_dataset(
                val_dir,
                spec.processed_directory,
                preprocessors,
                postprocessors,
            )
            sampler = ConfigCrossoverBatchSampler(
                groups=ds.idx_groups,
                batch_size=spec.batch_size,
                shuffle_groups=False,
                shuffle_within_groups=False,
                out_of_config_crossover_prob=0.0,
            )
            loader = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=spec.num_workers,
                batch_sampler=sampler,
            )
            identifier = f"{ds_type}-{ds_subtype}"
            eval_loaders[identifier] = loader

    pooling = GLOBAL_POOLINGS[spec.pooling]

    # Infer the number of features from the first batch
    num_features = train_dataset.get(0).x.shape[1]
    num_global_features = train_dataset.get(0).global_features.shape[0]

    model = GraphMLP(
        graph_input_dim=num_features,
        global_features_dim=num_global_features,
        graph_channels=spec.graph_channels,
        graph_layers=spec.graph_layers,
        linear_channels=spec.linear_channels,
        linear_layers=spec.linear_layers,
        dropout=spec.dropout,
        pooling_fn=pooling,
        pooling_feature_multiplier=spec.pooling_feature_multiplier,
        graph_conv=spec.graph_convolution_type,
        graph_conv_kwargs=spec.graph_convolution_kwargs,
    )
    model = torch_geometric.compile(model)  # type: ignore
    model = cast(GraphMLP, model)  # technically no, but it's fine

    criterion = CRITERIONS[spec.criterion](**spec.criterion_kwargs)
    optimizer = OPTIMIZERS[spec.optimizer](model.parameters(), **spec.optimizer_kwargs)
    scheduler = (
        SCHEDULERS[spec.scheduler](
            optimizer,
            epochs=spec.epochs,  # type: ignore
            steps_per_epoch=len(train_loader),  # type: ignore
            **spec.scheduler_kwargs,
        )
        if spec.scheduler
        else None
    )

    run_config = RunConfig(
        log_interval=spec.log_interval,
        log_table_interval=spec.log_table_interval,
        eval_interval=spec.eval_interval,
        eval_iterations=spec.eval_iterations,
        epochs=spec.epochs,
        save_checkpoints=spec.save_checkpoints,
    )

    return RunData(
        train_loader=train_loader,
        eval_loaders=eval_loaders,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        run_config=run_config,
        scheduler=scheduler,
    )


def get_dataset_dir(
    dataset_type: DatasetType, dataset_subtype: DatasetSubtype, split: str
) -> str:
    """Return the dataset directory for a specific dataset type and subtype."""
    return os.path.join(f"data/layout", dataset_type, dataset_subtype, split)


def generate_dataset_dirs(
    dataset_types: list[DatasetType],
    dataset_subtypes: list[DatasetSubtype],
    split: Literal["train", "val", "test"] = "train",
) -> list[str]:
    """Return a list of dataset directories for training and validation."""
    datasets: list[str] = []
    for dataset_type in dataset_types:
        for dataset_subtype in dataset_subtypes:
            datasets.append(get_dataset_dir(dataset_type, dataset_subtype, split=split))
    return datasets


def build_processors(processor_spec: ProcessorSpec) -> LayoutTransforms:
    graph_processor = (
        GRAPH_PROCESSORS[processor_spec.graph](**processor_spec.graph_kwargs)
        if processor_spec.graph is not None
        else None
    )

    node_processor = (
        NODE_PROCESSORS[processor_spec.node](**processor_spec.node_kwargs)
        if processor_spec.node is not None
        else None
    )

    config_processor = (
        CONFIG_PROCESSORS[processor_spec.config](**processor_spec.config_kwargs)
        if processor_spec.config is not None
        else None
    )

    opcode_processor = (
        OPCODE_PROCESSORS[processor_spec.opcode](**processor_spec.opcode_kwargs)
        if processor_spec.opcode is not None
        else None
    )

    global_processor = (
        GLOBAL_PROCESSORS[processor_spec.global_](**processor_spec.global_kwargs)
        if processor_spec.global_ is not None
        else None
    )

    target_prepocessor = (
        TARGET_PROCESSORS[processor_spec.target](**processor_spec.target_kwargs)
        if processor_spec.target is not None
        else None
    )

    return LayoutTransforms(
        node_transform=node_processor,
        graph_transform=graph_processor,
        config_transform=config_processor,
        opcode_transform=opcode_processor,
        global_transform=global_processor,
        target_transform=target_prepocessor,
    )


def build_dataset(
    directory: str,
    processed_directory: str,
    preprocessors: LayoutTransforms,
    postprocessors: LayoutTransforms,
) -> LayoutDataset:
    return LayoutDataset(
        directories=[directory],
        mode="memmapped",
        processed_dir=processed_directory,
        pretransforms=preprocessors,
        posttransforms=postprocessors,
    )
