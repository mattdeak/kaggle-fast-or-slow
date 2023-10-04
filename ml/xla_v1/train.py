from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from xgboost import XGBRegressor

from lib.metrics import xla_slowdown_from_runtime_preds
from lib.transforms.node_sum_pooling_with_graph_features_v1 import get_data

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


def build_model(**hyperparameters: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        device="cuda",
        **hyperparameters,
    )


def train(
    model: XGBRegressor,
    data: pl.LazyFrame,
) -> dict[str, float]:
    """Train the model and return train metrics."""
    train = to_datasplit(data)
    log_y_train = np.log(train.y)

    model.fit(train.X, log_y_train)
    train_preds = model.predict(train.X)

    loss = np.sqrt(np.mean((log_y_train - train_preds) ** 2))
    train_slowdown = xla_slowdown_from_runtime_preds(
        train.file_ids, train.y, train_preds
    )
    return {"train_perf": train_slowdown, "train_loss": loss}


def validate(model: XGBRegressor, data: pl.LazyFrame) -> dict[str, float]:
    val = to_datasplit(data)
    val_preds = model.predict(val.X)
    log_y_val = np.log(val.y)

    val_loss = np.sqrt(np.mean((log_y_val - val_preds) ** 2))
    val_slowdown = xla_slowdown_from_runtime_preds(val.file_ids, val.y, val_preds)
    return {"val_perf": val_slowdown, "val_loss": val_loss}


# def generate_submissions(
#     model: XGBRegressor, lazy_data: pl.LazyFrame
# ) -> npt.NDArray[np.float_]:
#     data = lazy_data.collect()
#     X = data.drop("file_id").to_arrow()
#     preds = model.predict(X)
#
#     # Get the top 5 predictions for each file_id
#     ranks = (
#         pl.DataFrame(
#             {
#                 "file_id": data["file_id"].to_list(),
#                 "pred": preds,
#             }
#         )
#         .groupby("file_id", maintain_order=True)
#         .agg(pl.col("pred").rank(method="ordinal"))
#     )
#
#     wandb.log({"test_perf": test_slowdown})
