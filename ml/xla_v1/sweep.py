import numpy as np

SWEEP_CONFIGURATION = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_perf"},
    "parameters": {
        "learning_rate": {"values": [0.0001, 0.001, 0.01, 0.05, 0.1]},
        "max_depth": {"values": np.arange(3, 14, 1).tolist()},
        "min_child_weight": {"values": np.arange(1, 6, 1).tolist()},
        "subsample": {"values": np.arange(0.5, 1.0, 0.1).tolist()},
        "colsample_bytree": {"values": np.arange(0.5, 1.0, 0.1).tolist()},
        "gamma": {"values": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        "n_estimators": {"values": [100, 200, 300, 400, 500]},
    },
}

DEFAULT_CONFIGURATION = {
    "learning_rate": 0.01,
    "max_depth": 3,
    "min_child_weight": 1,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "gamma": 0,
    "n_estimators": 100,
}
