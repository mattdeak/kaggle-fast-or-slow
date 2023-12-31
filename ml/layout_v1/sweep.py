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
        "graph_channels": {"values": [128, 256, 512]},
        "linear_layers": {"values": [0, 1, 2]},
        "linear_channels": {"values": [64, 128, 256]},
        "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
        "pooling": {"values": ["mean", "max", "multi"]},
        "graph_convolution_kwargs": {
            "parameters": {
                "heads": {"values": [1, 2, 4, 8]},
            },
        },
        "preprocessors": {
            "parameters": {
                "graph_kwargs": {
                    "parameters": {
                        "hops": {"value": 3},  # takes a while, but seems to be the best
                    },
                },
                "opcode": {"value": None},
            },
        },
        "postprocessors": {
            "parameters": {
                "opcode": {"values": ["ohe", "group-ohe-embedder"]},
            },
        },
        "graph_norm": {"values": ["graph", "layer"]},
        "linear_norm": {"values": ["layer", "batch"]},
        "optimizer": {"value": "adamw"},
        "optimizer_kwargs": {
            "parameters": {
                "lr": {"max": 5e-4, "min": 1e-4},
                "weight_decay": {"max": 0.01, "min": 0.0, "distribution": "uniform"},
            },
        },
        "epochs": {"value": 3},
        "batch_size": {"value": 16},
        "crossover": {"value": 0.0},
        "use_multi_edge": {"value": True},
        "alt_block": {"values": ["sage", "gat"]},  # main block is always gat
        "criterion": {"values": ["listMLE", "margin-loss"]},
        "criterion_kwargs": {
            "parameters": {
                "margin": {"min": 0.5, "max": 3.0},
                "n_permutations": {"value": 32},
            },
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
        "graph_channels": {"values": [128, 256, 512]},
        "linear_layers": {"values": [0, 1, 2]},
        "linear_channels": {"values": [64, 128, 256]},
        "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
        "pooling": {"values": ["mean", "max", "multi"]},
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
                "lr": {"max": 5e-4, "min": 1e-4},
                "weight_decay": {"max": 0.01, "min": 0.0, "distribution": "uniform"},
            },
        },
        "preprocessors": {
            "parameters": {
                "graph_kwargs": {
                    "parameters": {
                        "hops": {"value": 3},  # takes a while, but seems to be the best
                    },
                },
                "opcode": {"value": None},
            },
        },
        "postprocessors": {
            "parameters": {
                "opcode": {"values": ["ohe", "group-ohe-embedder"]},
            },
        },
        "epochs": {"value": 1},  # nlp is big, so we only do 1 epoch during a sweep
        "batch_size": {"value": 16},  # smaller cause bigger graphs
        "crossover": {"value": 0.0},
        "use_multi_edge": {"value": True},
        "alt_block": {"values": ["sage", "gat"]},  # main block is always gat
        "criterion": {"values": ["listMLE", "margin-loss"]},
        "criterion_kwargs": {
            "parameters": {
                "margin": {"min": 0.5, "max": 3.0},
                "n_permutations": {"value": 32},
            },
        },
    },
}


def define_sweeps() -> None:
    sweep_id1 = wandb.sweep(SWEEP_CONFIG_XLA_ONLY, project="kaggle-fast-or-slow")
    sweep_id2 = wandb.sweep(SWEEP_CONFIG_NLP_ONLY, project="kaggle-fast-or-slow")

    print(f"XLA-only sweep: {sweep_id1}")
    # print(f"NLP-only sweep: {sweep_id2}")


if __name__ == "__main__":
    define_sweeps()
