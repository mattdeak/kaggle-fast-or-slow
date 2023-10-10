SWEEP_CONFIGURATION = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_rmse"},
    "parameters": {
        "learning_rate": {"values": [0.0001, 0.001, 0.01]},
        "linear_layers": {
            "values": [
                [128, 64, 32],
                [256, 128, 64],
                [64, 64, 64, 64],
                [128, 128, 128, 128],
                [256, 256, 256, 256],
            ]
        },
        "graph_layers": {
            "values": [
                [64, 64, 64],
                [128, 128, 128],
                [256, 256, 256],
                [64, 64, 64, 64],
                [128, 128, 128, 128],
                [256, 256, 256, 256],
            ]
        },
    },
}
