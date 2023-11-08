import os
from dataclasses import dataclass
from typing import Any, Literal

import torch.nn as nn
from torch_geometric.loader import DataLoader

from ml.layout_v1.dataset import (ConcatenatedDataset, ConfigTransform,
                                  DataTransform, GlobalTransform,
                                  GraphTransform, LayoutDataset,
                                  LayoutTransforms, OpcodeEmbedder,
                                  TargetTransform)
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

XLA_DATA_DIR = "data/layout/xla"
NLP_DATA_DIR = "data/layout/nlp"

DatasetType = Literal["xla", "nlp"]
DatasetSubtype = Literal["default", "random"]

GraphProcessorName = Literal["config-communities"]
NodeProcessorName = Literal["node-processor"]
ConfigProcessorName = Literal["config-feature-generator"]
OpcodeProcessorName = Literal["group-ohe-embedder", "ohe"]
GlobalProcessorName = Literal["global-processor"]
TargetProcessorName = Literal["log"]

OptimizerName = Literal["adam", "sgd"]
SchedulerName = Literal["onecycle"]
CriterionName = Literal["listMLE", "margin-loss"]

GRAPH_PROCESSORS: dict[GraphProcessorName, type[GraphTransform]] = {
    "config-communities": ConfigNodeCommunityPreprocessor,
}

NODE_PROCESSORS: dict[NodeProcessorName, type[DataTransform]] = {
    "node-processor": NodeProcessor,
}

CONFIG_PROCESSORS: dict[ConfigProcessorName, type[ConfigTransform]] = {
    "config-feature-generator": ConfigFeatureGenerator,
}

OPCODE_PROCESSORS: dict[OpcodeProcessorName, type[OpcodeEmbedder]] = {
    "group-ohe-embedder": OpcodeGroupOHEEmbedder,
    "ohe": OpcodeOHEEmbedder,
}
GLOBAL_PROCESSORS: dict[GlobalProcessorName, type[GlobalTransform]] = {
    "global-processor": GlobalFeatureGenerator,
}
TARGET_PROCESSORS: dict[TargetProcessorName, type[TargetTransform]] = {
    "log": LogTargetTransform,
}

CRITERIONS: dict[CriterionName, nn.Module] = {
    "listMLE": listMLEalt,  # type: ignore this is a module but mypy doesn't like it
    "margin-loss": nn.MarginRankingLoss,
}
