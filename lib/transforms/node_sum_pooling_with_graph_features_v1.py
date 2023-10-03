import networkx as nx
import numpy as np
import polars as pl
from joblib.externals.cloudpickle.cloudpickle import Literal
from polars.type_aliases import IntoExpr

from lib.feature_name_mapping import NODE_FEATURE_MAP

# These columns are all 0 when you sum the entire train node dataset, so we choose to
# remove them.
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


def get_data(split: Literal["train", "valid", "test"] = "train") -> pl.LazyFrame:
    node_df = pl.scan_parquet(f"data/parquet/{split}/node/*.parquet", low_memory=True)
    config_df = pl.scan_parquet(
        f"data/parquet/{split}/config/*.parquet", low_memory=True
    )
    edge_df = pl.scan_parquet(f"data/parquet/{split}/edge/*.parquet", low_memory=True)

    # These columns are all 0 when you sum the entire train node dataset, so we can remove them
    # as we have no instances of them in the dataset
    # If they appear in the test set, we'll have to either ignore them or figure out a way

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

    graph_features = edge_df.group_by("file_id").map_groups(
        compute_graph_level_features,
        schema={
            "file_id": pl.Utf8,
            "average_degree": pl.Float32,
            "average_clustering": pl.Float32,
            "longest_path": pl.Float32,
        },
    )

    processed_config = (
        config_df.with_columns(
            (pl.col("config_runtime") / pl.col("config_runtime_normalizers")).alias(
                "label"
            )
        )
        .drop("config_runtime", "config_runtime_normalizers")
        .group_by("file_id")
        .map_groups(sample, schema=None)
    )

    # There are _a crazy amount_ of permutations for some files, so we'll randomly
    # sample no more than 1000 per file

    # |%%--%%| <4pXDbaJdoS|1dw5MUIyEX>
    processed_data = aggregations.join(
        graph_features, on="file_id", how="inner", validate="1:1"
    ).join(processed_config, on="file_id", how="inner", validate="1:m")

    return processed_data


def compute_graph_level_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute some simple graph-level statistics from an edge list.
    Assumes that the edge list is in the format of edge.parquet files."""
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


def sample(df: pl.DataFrame) -> pl.DataFrame:
    return df.sample(n=min(500, df.height))
