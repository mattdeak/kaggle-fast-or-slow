from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from lib.transforms.node_sum_pooling_with_graph_features_v1 import get_data

# |%%--%%| <1dw5MUIyEX|QdmDGzZ6w8>

node_df = pl.scan_parquet("data/parquet/train/node/*.parquet", low_memory=True)
unique_filenames = node_df.select("file_id").unique().collect().to_series()

import xgboost as xgb
# Alright we have our featureset and label. Let's try some basic linear regression first.
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# |%%--%%| <QdmDGzZ6w8|Rsmt6ubHAC>


class LazyFrameDataIterator(xgb.DataIter):
    def __init__(
        self, data: pl.LazyFrame, file_ids: pl.Series, file_batch_size: int = 1
    ):
        self.data = data
        self.cur = 0
        self.file_ids = file_ids
        self.max_ix = len(file_ids) // file_batch_size
        self.file_batch_size = file_batch_size

        super().__init__()

    @property
    def current_files(self):
        return self.file_ids[self.cur : self.cur + self.file_batch_size]

    def next(self, input_data: Callable[..., None]) -> int:
        """Get the next file id"""
        if self.cur >= self.max_ix:
            return 0
        else:
            data = self.data.filter(
                pl.col("file_id").is_in(self.current_files)
            ).collect()

            input_data(
                data=data.drop(["label", "file_id"]).to_numpy(),
                label=data.select("label").to_numpy(),
            )

            # Increment AFTER we get the data
            # you fucking moron (me)
            self.cur += self.file_batch_size
            return 1

    def reset(self):
        self.cur = 0

    def __iter__(self):
        return self


data = get_data()
train_files, test_files = train_test_split(
    unique_filenames, test_size=0.2, random_state=42
)

train = data.filter(pl.col("file_id").is_in(train_files))
test = data.filter(pl.col("file_id").is_in(test_files))

iter = LazyFrameDataIterator(train, file_ids=train_files, file_batch_size=500)
val_iter = LazyFrameDataIterator(test, file_ids=test_files, file_batch_size=500)

dmatrix = xgb.QuantileDMatrix(iter)
val_dmatrix = xgb.QuantileDMatrix(val_iter)

# |%%--%%| <Rsmt6ubHAC|tpftCdZ6FW>

param_dist = {
    "device": ["cuda"],
    "tree_method": ["gpu_hist"],
    "learning_rate": [0.0001, 0.001, 0.01, 0.05, 0.1],
    "max_depth": np.arange(3, 14, 1).tolist(),
    "min_child_weight": np.arange(1, 6, 1).tolist(),
    "subsample": np.arange(0.5, 1.0, 0.1).tolist(),
    "colsample_bytree": np.arange(0.5, 1.0, 0.1).tolist(),
    "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # "reg_alpha": [1e-5, 1e-2, 0.1, 1, 100],
    # "reg_lambda": [1e-5, 1e-2, 0.1, 1, 100],
    "num_estimators": [500],
}


# Because we can't use RandomizedSearchCV with pyarrow
# and we can't use xgb.cv with QuantileDMatrix
def randomly_choose_params(param_opts: dict[str, list[Any]]) -> dict[str, Any]:
    return {k: np.random.choice(v) for k, v in param_opts.items() if type(v) == list}


best_model = None
best_score = np.inf
best_params = None
for i in tqdm(range(50)):
    params = randomly_choose_params(param_dist)

    res = xgb.train(
        params,
        dmatrix,
        num_boost_round=100,
        evals=[(dmatrix, "train"), (val_dmatrix, "val")],
    )

    val_score = res.eval(val_dmatrix, "val")
    if (v := float(val_score.split(":")[1])) < best_score:
        best_score, best_model, best_params = v, res, params

val_preds = best_model.predict(val_dmatrix)
print(best_score)

pred_results = (
    data.filter(pl.col("file_id").is_in(test_files))
    .select("file_id", "label")
    .with_columns(
        preds=pl.Series(val_preds),
    )
).collect()
