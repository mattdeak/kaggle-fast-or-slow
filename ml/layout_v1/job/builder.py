import os
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch_geometric import torch_geometric
from torch_geometric.loader import DataLoader

from ml.layout_v1.dataset import (ConcatenatedDataset, DataTransform,
                                  LayoutDataset, LayoutTransforms)
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
    use_amp: bool = False


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
    train_data_dirs_list = [
        d
        for ds_subtypes in train_data_directories.values()
        for d in ds_subtypes.values()
    ]

    # Fit the preprocessors on the training data
    # before they can be used on the validation data
    if preprocessors.node_transform:
        if hasattr(preprocessors.node_transform, "fit"):
            preprocessors.node_transform = fit_node_processor(
                train_data_dirs_list, preprocessors.node_transform
            )

    if postprocessors.node_transform:
        if hasattr(postprocessors.node_transform, "fit"):
            postprocessors.node_transform = fit_node_processor(
                train_data_dirs_list, postprocessors.node_transform
            )

    # we have to readjust the global processors because they can depend
    # on the dataset type and subtype. We'll make copies of the processors
    # and then set the dataset type and subtype on them.
    # This is obviously bad, but who cares this code lasts for another week.
    train_datasets: list[LayoutDataset] = []
    for ds_type, ds_subtypes in train_data_directories.items():
        for ds_subtype, ds_dir in ds_subtypes.items():
            global_preprocessor = (
                GLOBAL_PROCESSORS[spec.preprocessors.global_](
                    **spec.preprocessors.global_kwargs,
                    dataset_type=ds_type,
                    dataset_subtype=ds_subtype,
                )
                if spec.preprocessors.global_
                else None
            )
            global_postprocessor = (
                GLOBAL_PROCESSORS[spec.postprocessors.global_](
                    **spec.postprocessors.global_kwargs,
                    dataset_type=ds_type,
                    dataset_subtype=ds_subtype,
                )
                if spec.postprocessors.global_
                else None
            )

            ds_preprocessors = None
            ds_postprocessors = None

            if spec.preprocessors.global_ is not None and global_preprocessor:
                ds_preprocessors = deepcopy(preprocessors)
                ds_preprocessors.global_transform = global_preprocessor

            if spec.postprocessors.global_ is not None and global_postprocessor:
                ds_postprocessors = deepcopy(postprocessors)
                ds_postprocessors.global_transform = global_postprocessor

            if ds_preprocessors is None:
                ds_preprocessors = preprocessors
            if ds_postprocessors is None:
                ds_postprocessors = postprocessors

            train_datasets.append(
                build_dataset(
                    ds_dir,
                    spec.processed_directory,
                    ds_preprocessors,
                    ds_postprocessors,
                )
            )
    train_dataset = ConcatenatedDataset(train_datasets)

    # Debug
    print("IDX Groups len: ", len(train_dataset.idx_groups))

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
            val_dir = get_dataset_dir(ds_type, ds_subtype, split="valid")
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
        graph_norm=spec.graph_norm,
        linear_norm=spec.linear_norm,
        use_multi_edge=spec.use_multi_edge,
        main_block=spec.main_block,
        alt_block=spec.alt_block,
    )
    # model = torch_geometric.compile(model)  # type: ignore
    model = cast(GraphMLP, model)  # technically no, but it's fine

    optimizer = OPTIMIZERS[spec.optimizer](model.parameters(), **spec.optimizer_kwargs)
    criterion = CRITERIONS[spec.criterion](**spec.criterion_kwargs)

    steps_per_epoch = len(train_loader)
    scheduler = (
        SCHEDULERS[spec.scheduler](
            optimizer,
            epochs=spec.epochs,  # type: ignore
            steps_per_epoch=steps_per_epoch,
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
        use_amp=spec.use_amp,
    )

    return RunData(
        train_loader=train_loader,
        eval_loaders=eval_loaders,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        run_config=run_config,
    )


def get_dataset_dir(
    dataset_type: DatasetType, dataset_subtype: DatasetSubtype, split: str
) -> str:
    """Return the dataset directory for a specific dataset type and subtype."""
    return os.path.join(f"data/layout", dataset_type, dataset_subtype, split)


def generate_dataset_dirs(
    dataset_types: list[DatasetType],
    dataset_subtypes: list[DatasetSubtype],
    split: Literal["train", "valid", "test"] = "train",
) -> dict[str, dict[str, str]]:
    """Return a list of dataset directories for training and validation."""
    datasets: dict[str, dict[str, str]] = defaultdict(dict)
    for dataset_type in dataset_types:
        for dataset_subtype in dataset_subtypes:
            datasets[dataset_type][dataset_subtype] = get_dataset_dir(
                dataset_type, dataset_subtype, split=split
            )

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
        global_transform=None,  # handled elsewhere
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
        multiprocess=False,
    )


def fit_node_processor(
    directories: list[str], processor: DataTransform
) -> DataTransform:
    """This isn't the smartest way to do this, but whatever who knows
    how long it'll even last.


    We're just going to manually read and stack all node features,
    then fit the processor on that.
    """
    assert hasattr(processor, "fit"), "Processor does not have a fit method"
    print("Fitting node processor")

    node_features: list[npt.NDArray[np.float_]] = []

    for directory in directories:
        files = os.listdir(directory)

        # load npz
        for f in files:
            if f.endswith(".npz"):
                d = np.load(os.path.join(directory, f))
                nf = d["node_feat"]
                node_features.append(nf)

    node_features = np.vstack(node_features)  # type: ignore

    processor.fit(node_features)
    return processor
