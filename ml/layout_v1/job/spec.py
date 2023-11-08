import os
from dataclasses import dataclass
from typing import Any, Literal

import torch.nn as nn
from pydantic import BaseModel
from torch_geometric.loader import DataLoader

from ml.layout_v1.dataset import (ConcatenatedDataset, ConfigTransform,
                                  DataTransform, GlobalTransform,
                                  GraphTransform, LayoutDataset,
                                  LayoutTransforms, OpcodeEmbedder,
                                  TargetTransform)
from ml.layout_v1.job.constants import (ConfigProcessorName, DatasetSubtype,
                                        DatasetType, GlobalProcessorName,
                                        GraphProcessorName, NodeProcessorName,
                                        OpcodeProcessorName, OptimizerName,
                                        SchedulerName, TargetProcessorName)
from ml.layout_v1.losses import listMLEalt
from ml.layout_v1.preprocessors.config_preproc import ConfigFeatureGenerator
from ml.layout_v1.preprocessors.global_preproc import GlobalFeatureGenerator
from ml.layout_v1.preprocessors.graph_preproc import \
    ConfigNodeCommunityPreprocessor
from ml.layout_v1.preprocessors.node_preproc import NodeProcessor
from ml.layout_v1.preprocessors.opcode_preproc import (OpcodeGroupOHEEmbedder,
                                                       OpcodeOHEEmbedder)
from ml.layout_v1.preprocessors.target_preproc import LogTargetTransform
from ml.layout_v1.sampler import ConfigCrossoverBatchSampler


class ProcessorSpec(BaseModel):
    graph: GraphProcessorName | None = "config-communities"
    graph_kwargs: dict[str, Any] = {"hops": 2}

    node: NodeProcessorName | None = "node-processor"
    node_kwargs: dict[str, Any] = {}

    config: ConfigProcessorName | None = "config-feature-generator"
    config_kwargs: dict[str, Any] = {}

    opcode: OpcodeProcessorName | None = "group-ohe-embedder"
    opcode_kwargs: dict[str, Any] = {}

    global_: GlobalProcessorName | None = "global-processor"
    global_kwargs: dict[str, Any] = {}

    target: TargetProcessorName | None = "log"
    target_kwargs: dict[str, Any] = {}


class JobSpec(BaseModel):
    """Configuration for a training/evaluation job.


    This has to be configurable via basic data types so we can
    launch it with wandb sweeps. Lookup tables are handled by
    the `generate` method.
    """

    # Dataset
    dataset_types: list[DatasetType] = ["xla"]
    dataset_subtypes: list[DatasetSubtype] = ["default", "random"]
    processed_directory: str = "data/processed"

    # Training
    log_interval: int = 1000
    log_table_interval: int = 20000
    eval_interval: int = 10000
    eval_iterations: int = 512
    epochs: int = 6

    # Model Config
    graph_layers: int = 3
    graph_channels: int = 128
    linear_layers: int = 3
    linear_channels: int = 128
    dropout: float = 0.0
    pooling: Literal["sum", "mean", "max", "multi"] = "multi"

    # These need to be typed better
    graph_convolution_type: Literal["sage", "gat"] = "gat"
    graph_convolution_kwargs: dict[str, Any] = {"heads": 4}

    # training
    batch_size: int = 16
    num_workers: int = 4
    use_amp: bool = False
    wandb: bool = True
    save_checkpoints: bool = True
    max_checkpoints: int = 5

    # optimizer
    optimizer: OptimizerName = "adamw"
    optimizer_kwargs: dict[str, Any] = {"lr": 3e-4, "weight_decay": 0.01}

    criterion: Literal["listMLE", "margin-loss"] = "margin-loss"
    criterion_kwargs: dict[str, Any] = {"margin": 1.0}

    # scheduler
    scheduler: SchedulerName | None = "onecycle"
    scheduler_kwargs: dict[str, Any] = {"max_lr": 0.01}

    # processors
    preprocessors: ProcessorSpec = ProcessorSpec()
    postprocessors: ProcessorSpec = ProcessorSpec(
        graph=None, node=None, config=None, opcode=None, global_=None, target=None
    )

    # crossover
    crossover: float = 0.0

    @property
    def pooling_feature_multiplier(self) -> int:
        return 4 if self.pooling == "multi" else 1
