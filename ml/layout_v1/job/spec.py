from typing import Any, Literal

from pydantic import BaseModel, model_validator

from ml.layout_v1.job.constants import (ConfigProcessorName, DatasetSubtype,
                                        DatasetType, GlobalProcessorName,
                                        GraphProcessorName, NodeProcessorName,
                                        OpcodeProcessorName, OptimizerName,
                                        SchedulerName, TargetProcessorName)


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

    target: TargetProcessorName | None = None
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

    graph_norm: Literal["graph", "layer"] = "graph"
    linear_norm: Literal["layer", "batch"] = "layer"

    # Multiple Edge Indices
    use_multi_edge: bool = False

    main_block: Literal["gat", "sage"] = "gat"
    alt_block: Literal["gat", "sage"] = "sage"

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
    scheduler_kwargs: dict[str, Any] = {"max_lr": 0.01, "pct_start": 0.1}

    # processors
    preprocessors: ProcessorSpec = ProcessorSpec()
    postprocessors: ProcessorSpec = ProcessorSpec(
        graph=None, node=None, config=None, opcode=None, global_=None, target=None
    )

    use_distribution_flag: bool = True

    # crossover
    crossover: float = 0.0

    @property
    def pooling_feature_multiplier(self) -> int:
        return 4 if self.pooling == "multi" else 1

    @model_validator(mode="before")
    @classmethod
    def if_not_gat_coerce_heads(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Coerce the graph convolution kwargs to be empty if not using GAT."""
        if values["graph_convolution_type"] != "gat":
            values["graph_convolution_kwargs"] = {}
        return values

    @model_validator(mode="before")
    @classmethod
    def if_not_margin_loss_then_del_criterion_kwargs(
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Coerce the use_multi_edge flag to False if not using multi pooling."""
        if values["criterion"] != "margin-loss":
            values["criterion_kwargs"] = {}

        return values
