SWEEP_CONFIGURATION = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_rmse"},
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-2},
        "linear_layers": {
            "values": [
                [256, 256],
                [256, 512],
                [512, 1024],
                [1024, 2048],
                [2048, 4096],
                [4096, 8192],
            ]
        },
        "graph_layers": {
            "values": [[64, 128], [128, 256, 512], [256, 512, 1024], [512, 1024, 2048]]
        },
    },
}
