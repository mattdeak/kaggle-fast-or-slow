import polars as pl

from lib.transforms.node_sum_pooling_with_graph_features_v1 import get_data

# |%%--%%| <1dw5MUIyEX|QdmDGzZ6w8>
split = "train"

node_df = pl.scan_parquet(f"data/parquet/{split}/node/*.parquet", low_memory=True)
config_df = pl.scan_parquet(f"data/parquet/{split}/config/*.parquet", low_memory=True)
edge_df = pl.scan_parquet(f"data/parquet/{split}/edge/*.parquet", low_memory=True)


# |%%--%%| <QdmDGzZ6w8|Rsmt6ubHAC>

nodes = node_df.collect()
nodes.group_by("file_id").agg(pl.col("element_size_in_bits").sum())

edges = edge_df.collect()

# |%%--%%| <Rsmt6ubHAC|Lq1o756Kkm>

import seaborn as sns

sns.histplot(nodes["node_opcode"].value_counts())
