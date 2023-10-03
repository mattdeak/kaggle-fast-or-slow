import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import wandb
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from lib.metrics import xla_slowdown_from_runtime_preds
from lib.transforms.node_sum_pooling_with_graph_features_v1 import get_data

PARAM_DIST = {
    "device": ["cuda"],
    "tree_method": ["gpu_hist"],
    "learning_rate": [0.0001, 0.001, 0.01, 0.05, 0.1],
    "max_depth": np.arange(3, 14, 1).tolist(),
    "min_child_weight": np.arange(1, 6, 1).tolist(),
    "subsample": np.arange(0.5, 1.0, 0.1).tolist(),
    "colsample_bytree": np.arange(0.5, 1.0, 0.1).tolist(),
    "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "n_estimators": [100, 200, 300, 400, 500],
}


def to_X_and_y(
    df: pl.LazyFrame,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    X = df.drop(["file_id", "label"]).collect().to_numpy()
    y = df.select("label").collect().to_numpy().ravel()
    return X, y


def main(sample: bool = False):
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

    train_file_ids = train_data.select("file_id").collect()["file_id"]
    val_file_ids = val_data.select("file_id").collect()["file_id"]
    X_train, y_train = to_X_and_y(train_data)

    print("Size of training data:", X_train.shape)

    log_y_train = np.log(y_train)

    reg = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_jobs=1,
        verbosity=1,
    )

    clf = RandomizedSearchCV(
        reg,
        param_distributions=PARAM_DIST,
        n_iter=50,
    )

    clf.fit(X_train, log_y_train)

    for k, v in clf.best_params_.items():
        wandb.config[k] = v

    train_preds = clf.predict(X_train)

    del X_train
    train_slowdown = xla_slowdown_from_runtime_preds(
        train_file_ids, y_train, train_preds
    )

    del y_train, train_preds
    wandb.log({"train_slowdown": train_slowdown})

    X_val, y_val = to_X_and_y(val_data)

    val_preds = clf.predict(X_val)
    y_pred = clf.predict(X_val)

    val_slowdown = xla_slowdown_from_runtime_preds(val_file_ids, y_val, val_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to sample the data. Useful for debugging.",
    )
    args = parser.parse_args()
    main(sample=args.sample)
