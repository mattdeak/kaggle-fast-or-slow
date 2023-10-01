from collections.abc import Callable
from typing import Any

import networkx as nx
import numpy as np
import polars as pl
from polars.type_aliases import IntoExpr

from lib.feature_name_mapping import NODE_FEATURE_MAP

# |%%--%%| <THmYMuHfiW|4kHIdeAYiO>
node_df = pl.scan_parquet("data/parquet/train/node/*.parquet", low_memory=True)
config_df = pl.scan_parquet("data/parquet/train/config/*.parquet", low_memory=True)
edge_df = pl.scan_parquet("data/parquet/train/edge/*.parquet", low_memory=True)

unique_files = node_df.select("file_id").unique().sort("file_id").collect()
# Testing
# node_df = pl.scan_parquet(
#     "data/parquet/train/node/xception_imagenet_4f53c8f4f490d39d.npz.parquet"
# )
# config_df = pl.scan_parquet(
#     "data/parquet/train/config/xception_imagenet_4f53c8f4f490d39d.npz.parquet"
# )
# edge_df = pl.scan_parquet(
#     "data/parquet/train/edge/xception_imagenet_4f53c8f4f490d39d.npz.parquet"
# )

# |%%--%%| <4kHIdeAYiO|7oe3mQzQUO>

# These columns are all 0 when you sum the entire train node dataset, so we can remove them
# as we have no instances of them in the dataset
# If they appear in the test set, we'll have to either ignore them or figure out a way

DEAD_COLS = [
    "element_size_in_bits",
    "shape_element_type_is_invalid_type",
    "shape_element_type_is_s8",
    "shape_element_type_is_s16",
    "shape_element_type_is_s64",
    "shape_element_type_is_u64",
    "shape_element_type_is_f16",
    "shape_element_type_is_f64",
    "shape_element_type_is_c64",
    "shape_element_type_is_c128",
    "shape_element_type_is_opaque_type",
    "shape_element_type_is_token",
    "dimensions_4",
    "dimensions_5",
    "window_size_3",
    "window_size_4",
    "window_size_5",
    "window_stride_3",
    "window_stride_4",
    "window_stride_5",
    "window_padding_low_3",
    "window_padding_low_4",
    "window_padding_low_5",
    "window_padding_high_3",
    "window_padding_high_4",
    "window_padding_high_5",
    "window_window_dilation_3",
    "window_window_dilation_4",
    "window_window_dilation_5",
    "window_base_dilation_3",
    "window_base_dilation_4",
    "window_base_dilation_5",
    "window_window_reversal_3",
    "window_window_reversal_4",
    "window_window_reversal_5",
    "convolution_dim_numbers_input_spatial_dims_3",
    "convolution_dim_numbers_kernel_spatial_dims_3",
    "is_stable",
]

ohe_cols: list[IntoExpr] = []

for key in NODE_FEATURE_MAP.values():  # The str
    col = (
        pl.when(pl.col("node_opcode") == key)
        .then(True)
        .otherwise(False)
        .alias(f"opcode_{key}")
    )
    ohe_cols.append(col)

processed = node_df.drop(DEAD_COLS).with_columns(ohe_cols)

# Just sum up all columns over node id
aggregations = (
    processed.select(pl.exclude("node_opcode", "node_id")).group_by("file_id").sum()
)

# Some columns are all 0, so we can filter them out

# Calculate some basic graph-level statistics

# |%%--%%| <7oe3mQzQUO|RMevGpafOk>


def compute_graph_level_features(df: pl.DataFrame) -> pl.DataFrame:
    # Convert to tuples
    to = df["to"].cast(pl.Int32).to_list()
    frm = df["from"].cast(pl.Int32).to_list()

    edgelist = [(t, f) for t, f in zip(to, frm)]

    graph = nx.DiGraph(edgelist)
    average_degree = np.mean([d for _, d in graph.degree()])
    average_clustering = nx.average_clustering(graph)
    longest_path = nx.dag_longest_path_length(graph)

    return pl.DataFrame(
        {
            "file_id": df["file_id"][0],
            "average_degree": average_degree,
            "average_clustering": average_clustering,
            "longest_path": longest_path,
        },
    )


graph_features = edge_df.group_by("file_id").map_groups(
    compute_graph_level_features,
    schema={
        "file_id": pl.Utf8,
        "average_degree": pl.Float32,
        "average_clustering": pl.Float32,
        "longest_path": pl.Float32,
    },
)


# |%%--%%| <RMevGpafOk|4pXDbaJdoS>


def sample(df: pl.DataFrame) -> pl.DataFrame:
    return df.sample(n=min(500, df.height))


processed_config = (
    config_df.with_columns(
        (pl.col("config_runtime") / pl.col("config_runtime_normalizers")).alias("label")
    )
    .drop("config_runtime", "config_runtime_normalizers")
    .group_by("file_id")
    .map_groups(sample, schema=None)
)


# There are _a crazy amount_ of permutations for some files, so we'll randomly
# sample no more than 1000 per file


# |%%--%%| <4pXDbaJdoS|1dw5MUIyEX>
feature_df_lazy = aggregations.join(
    graph_features, on="file_id", how="inner", validate="1:1"
).join(processed_config, on="file_id", how="inner", validate="1:m")


# |%%--%%| <1dw5MUIyEX|QdmDGzZ6w8>


import xgboost as xgb
# Alright we have our featureset and label. Let's try some basic linear regression first.
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from tqdm.auto import tqdm
from xgboost import XGBRegressor

reg = XGBRegressor(
    n_estimators=100,
    objective="reg:squarederror",
    n_jobs=1,
)


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
        if self.cur > self.max_ix:
            return 0
        else:
            self.cur += self.file_batch_size
            input_data(
                data=self.data.drop(["label", "file_id"])
                .filter(pl.col("file_id").is_in(self.current_files))
                .collect()
                .to_numpy(),
                label=self.data.filter(pl.col("file_id").is_in(self.current_files))
                .select("label")
                .collect()
                .to_numpy(),
            )
            return 1

    def reset(self):
        self.cur = 0

    def __iter__(self):
        return self


train_files, test_files = train_test_split(
    unique_files.to_series(), test_size=0.2, random_state=42
)

iter = LazyFrameDataIterator(feature_df_lazy, file_ids=train_files, file_batch_size=500)
val_iter = LazyFrameDataIterator(
    feature_df_lazy, file_ids=test_files, file_batch_size=500
)

dmatrix = xgb.QuantileDMatrix(iter)
val_dmatrix = xgb.QuantileDMatrix(val_iter)

param_dist = {
    "device": ["cuda"],
    "tree_method": ["gpu_hist"],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "max_depth": np.arange(3, 10, 1).tolist(),
    "min_child_weight": np.arange(1, 6, 1).tolist(),
    "subsample": np.arange(0.5, 1.0, 0.1).tolist(),
    "colsample_bytree": np.arange(0.5, 1.0, 0.1).tolist(),
    "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "reg_alpha": [1e-5, 1e-2, 0.1, 1, 100],
    "reg_lambda": [1e-5, 1e-2, 0.1, 1, 100],
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
        callbacks=[
            xgb.callback.EvaluationMonitor(show_stdv=False),
            xgb.callback.EarlyStopping(3),
        ],
    )

    val_score = res.eval(val_dmatrix, "val")
    if (v := float(val_score.split(":")[1])) < best_score:
        best_score = v
        best_model = res
        best_params = params

best_model.save_model("models/xgb_xla.model")
print(best_score)
