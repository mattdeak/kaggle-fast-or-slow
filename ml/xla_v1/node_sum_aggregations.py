import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow
import wandb
from xgboost import XGBRegressor

from lib.metrics import xla_slowdown_from_runtime_preds
from lib.transforms.node_sum_pooling_with_graph_features_v1 import get_data

sweep_configuration = {
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

ArrowTable = Any


@dataclass
class DataSplit:
    X: ArrowTable
    y: npt.NDArray[np.float_]
    file_ids: list[str]


def to_datasplit(
    df: pl.LazyFrame,
) -> DataSplit:
    data = df.collect()
    X = data.drop(["file_id", "label"]).to_arrow()
    y = data["label"].to_numpy().ravel()
    file_ids = data["file_id"].to_list()
    return DataSplit(X, y, file_ids)


def main(*, sample: bool = False, **hyperparameters: dict[str, Any]):
    run = wandb.init(
        project="kaggle-fast-or-slow",
        config={
            "data_pipeline": "node_sum_pooling_with_graph_features_v1",
            "model": "xgboost_regressor",
        },
        notes="This approach simply sums all node features for the graph aggregation. We compute a few basic graph features, and then join on the config features. XGBoost is tuned on a log-transformed runtime value.",
        job_type="train_and_eval",
        mode="disabled" if sample else "online",  # don't record in sample mode
    )

    train_data = get_data("train")
    val_data = get_data("valid")

    if sample:
        train_data = train_data.limit(1000)
        val_data = val_data.limit(1000)

    train = to_datasplit(train_data)

    log_y_train = np.log(train.y)
    reg = XGBRegressor(
        objective="reg:squarederror",
        device="cuda",
        **hyperparameters,
    )

    reg.fit(train.X, log_y_train)
    train_preds = reg.predict(train.X)

    train_slowdown = xla_slowdown_from_runtime_preds(
        train.file_ids, train.y, train_preds
    )

    del train
    wandb.log({"train_perf": train_slowdown})

    val = to_datasplit(val_data)

    val_preds = reg.predict(val.X)
    val_slowdown = xla_slowdown_from_runtime_preds(val.file_ids, val.y, val_preds)
    wandb.log({"val_perf": val_slowdown})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    if args.sample:
        main(sample=True)
    elif args.sweep:
        sweep_id = wandb.sweep(sweep_configuration, project="kaggle-fast-or-slow")
        wandb.agent(sweep_id, function=main)
