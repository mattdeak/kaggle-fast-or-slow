import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# |%%--%%| <L7JTWoKk0f|F4oY4e2xGz>
# 📁 Define a Function to Load DataFrames
# This function loads data stored in different splits (train, valid, test) from a specified directory.
# It reads files in the directory, extracts data using NumPy, and organizes it into DataFrames.


def load_df(directory: str) -> dict[str, pd.DataFrame]:
    splits: list[str] = ["train", "valid", "test"]
    dfs: dict[str, pd.DataFrame] = {}

    for split in splits:
        path = os.path.join(directory, split)
        files = os.listdir(path)
        list_df: list[dict[str, Any]] = []

        for file in files:
            d: dict[str, Any] = dict(np.load(os.path.join(path, file)))
            d["file"] = file
            list_df.append(d)

        dfs[split] = pd.DataFrame.from_dict(list_df)  # type: ignore

    return dfs


# |%%--%%| <F4oY4e2xGz|FiilGRP5Xg>

# 📄 Load data using the defined function and store it in the 'tile_xla' variable
tile_xla = load_df("data/npz/tile/xla/")
