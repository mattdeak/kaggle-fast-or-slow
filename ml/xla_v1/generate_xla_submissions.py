"""This is a sandboxy script to generate submissions using the XLA tile model."""
import tempfile

import numpy as np
import polars as pl
import xgboost

import wandb
from lib.transforms.node_sum_pooling_with_graph_features_v1 import get_data

MODEL_VERSION = "v0"
MODEL_ID = "xla_tile_xgb"

api = wandb.Api()
artifact = api.artifact(f"kaggle-fast-or-slow/{MODEL_ID}:{MODEL_VERSION}", type="model")

with tempfile.TemporaryDirectory() as tmpdir:
    artifact.download(root=tmpdir)
    model = xgboost.XGBRegressor()
    model.load_model(f"{tmpdir}/{MODEL_ID}.xgb")

test = get_data("test")

df = test.collect()
file_ids = df["file_id"]
X = df.drop(["file_id", "label"])

del df

results = model.predict(X)

df = pl.DataFrame({"file_id": file_ids, "prediction": results})

# |%%--%%| <YA0HnVIoLc|OWh4ScZRyt>
ranked = df.groupby("file_id", maintain_order=True).agg(
    pl.col("prediction").rank(method="ordinal").alias("rank")
)

sorted = ranked.with_columns(
    pl.col("rank").apply(lambda x: np.argsort(x)[:5]).alias("sorted")
).select(
    pl.col("sorted").apply(lambda x: ";".join([str(i) for i in x])).alias("TopConfigs"),
    ("tile:xla:" + pl.col("file_id").str.replace(".npz", "")).alias("ID"),
)


submissions_file = pl.read_csv("data/sample_submission.csv")
new_submissions_file = submissions_file.join(sorted, on="ID", how="left")

new_submissions_file = new_submissions_file.with_columns(
    pl.col("TopConfigs_right").fill_null(pl.col("TopConfigs")).alias("TopConfigs")
).select("ID", "TopConfigs")


# |%%--%%| <OWh4ScZRyt|KeQExirgFh>
new_submissions_file.write_csv(f"data/{MODEL_ID}_{MODEL_VERSION}.csv", has_header=True)
