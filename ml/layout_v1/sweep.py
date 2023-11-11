import wandb

# from ml.layout_v1.train import DEFAULT_CONFIG, run

SWEEP_CONFIG_XLA_ONLY = {
    "method": "random",
    "name": "hp-sweep",
    "metric": {"name": "full/avg/kendall_tau", "goal": "maximize"},
    "parameters": {
        "dataset_types": {"value": ["xla"]},
        "dataset_subtypes": {"value": ["default", "random"]},
        "graph_layers": {"values": [3, 4, 5]},
        "graph_channels": {"values": [64, 128, 256]},
        "linear_layers": {"values": [2, 3, 4, 5]},
        "linear_channels": {"values": [64, 128, 256]},
        "dropout": {"values": [0.0, 0.05, 0.1, 0.2]},
        "pooling": {"values": ["mean", "max", "multi"]},
        "graph_convolution_type": {"values": ["sage", "gat"]},
        "graph_convolution_kwargs": {
            "parameters": {
                "heads": {"values": [1, 2, 4, 8]},
            },
        },
        "graph_norm": {"values": ["graph", "layer"]},
        "linear_norm": {"values": ["layer", "batch"]},
        "optimizer": {"value": ["adamw"]},
        "optimizer_kwargs": {
            "parameters": {
                "lr": {"max": 5e-3, "min": 1e-4},
                "weight_decay": {"max": 0.01, "min": 0.0, "distribution": "uniform"},
            },
        },
        "epochs": {"values": [6]},
        "batch_size": {"values": [8, 16]},
        "preprocessors": {
            "parameters": {
                "global_kwargs": {
                    "parameters": {
                        "subtype_indicator": {"values": [True, False]},
                    },
                },
            }
        },
        "crossover": {"min": 0.0, "max": 0.2},
        "use_distribution_flag": {"values": [True, False]},
        "use_multi_edge": {"values": [True, False]},
        "criterion": {"values": ["listMLE", "margin-loss"]},
        "criterion_kwargs": {
            "parameters": {"margin": {"min": 0.5, "max": 3.0}},
        },
    },
}


def define_sweep() -> str:
    sweep_id = wandb.sweep(SWEEP_CONFIG_XLA_ONLY, project="kaggle-fast-or-slow")
    return sweep_id


if __name__ == "__main__":
    id_ = define_sweep()
    print(id_)
