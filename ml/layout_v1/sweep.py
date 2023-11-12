import wandb

PROGRAM_PATH = "ml/layout_v1/train.py"

# from ml.layout_v1.train import DEFAULT_CONFIG, run

SWEEP_CONFIG_XLA_ONLY = {
    "program": PROGRAM_PATH,
    "command": ["${env}", "${interpreter}", "${program}"],
    "method": "random",
    "name": "hp-sweep-xla-only",
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
        "optimizer": {"value": "adamw"},
        "optimizer_kwargs": {
            "parameters": {
                "lr": {"max": 5e-3, "min": 1e-4},
                "weight_decay": {"max": 0.01, "min": 0.0, "distribution": "uniform"},
            },
        },
        "epochs": {"value": 6},
        "batch_size": {"values": [8, 16]},
        "crossover": {"min": 0.0, "max": 0.2},
        "use_distribution_flag": {"values": [True, False]},
        "use_multi_edge": {"values": [True, False]},
        "criterion": {"values": ["listMLE", "margin-loss"]},
        "criterion_kwargs": {
            "parameters": {"margin": {"min": 0.5, "max": 3.0}},
        },
    },
}

SWEEP_CONFIG_NLP_ONLY = {
    "program": PROGRAM_PATH,
    "command": ["${env}", "${interpreter}", "${program}"],
    "method": "random",
    "name": "hp-sweep-nlp-only",
    "metric": {"name": "full/avg/kendall_tau", "goal": "maximize"},
    "parameters": {
        "dataset_types": {"value": ["nlp"]},
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
        "optimizer": {"value": "adamw"},
        "optimizer_kwargs": {
            "parameters": {
                "lr": {"max": 5e-3, "min": 1e-4},
                "weight_decay": {"max": 0.01, "min": 0.0, "distribution": "uniform"},
            },
        },
        "epochs": {"value": 6},
        "batch_size": {"values": [8, 16]},
        "crossover": {"min": 0.0, "max": 0.2},
        "use_distribution_flag": {"values": [True, False]},
        "use_multi_edge": {"values": [True, False]},
        "criterion": {"values": ["listMLE", "margin-loss"]},
        "criterion_kwargs": {
            "parameters": {"margin": {"min": 0.5, "max": 3.0}},
        },
    },
}

SWEEP_CONFIG_ALL_DATA = {
    "program": PROGRAM_PATH,
    "command": ["${env}", "${interpreter}", "${program}"],
    "method": "random",
    "name": "hp-sweep-all-data",
    "metric": {"name": "full/avg/kendall_tau", "goal": "maximize"},
    "parameters": {
        "dataset_types": {"value": ["nlp", "xla"]},
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
        "optimizer": {"value": "adamw"},
        "optimizer_kwargs": {
            "parameters": {
                "lr": {"max": 5e-3, "min": 1e-4},
                "weight_decay": {"max": 0.01, "min": 0.0, "distribution": "uniform"},
            },
        },
        "epochs": {"value": 6},
        "batch_size": {"values": [8, 16]},
        "crossover": {"min": 0.0, "max": 0.2},
        "use_distribution_flag": {"values": [True, False]},
        "use_multi_edge": {"values": [True, False]},
        "criterion": {"values": ["listMLE", "margin-loss"]},
        "criterion_kwargs": {
            "parameters": {"margin": {"min": 0.5, "max": 3.0}},
        },
    },
}


def define_sweeps() -> None:
    sweep_id1 = wandb.sweep(SWEEP_CONFIG_XLA_ONLY, project="kaggle-fast-or-slow")
    sweep_id2 = wandb.sweep(SWEEP_CONFIG_NLP_ONLY, project="kaggle-fast-or-slow")
    sweep_id3 = wandb.sweep(SWEEP_CONFIG_ALL_DATA, project="kaggle-fast-or-slow")

    print(f"XLA-only sweep: {sweep_id1}")
    print(f"NLP-only sweep: {sweep_id2}")
    print(f"All-data sweep: {sweep_id3}")


if __name__ == "__main__":
    define_sweeps()
