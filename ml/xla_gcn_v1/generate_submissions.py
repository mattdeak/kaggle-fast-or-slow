import os

import numpy as np
import torch

import wandb
from ml.xla_gcn_v1.dataset import parse_file
from ml.xla_gcn_v1.sagemlp import SAGEMLP

DATA_DIR = "data/npz/tile/xla/test"
INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".npz")]

test_file = files[0]
# |%%--%%| <ViXrO6qwi1|O6LNSSPtuo>
RUN_ID = "kaggle-fast-or-slow/8tthuo5u"
WEIGHTS_PATH = "models/8tthuo5u/100000.pt"

api = wandb.Api()
run = api.run(RUN_ID)

config = run.config


# |%%--%%| <O6LNSSPtuo|P7z3qEAf51>

model = SAGEMLP(
    graph_input_dim=INPUT_DIM,
    global_input_dim=GLOBAL_INPUT_DIM,
    sage_channels=config["sage_channels"],
    sage_layers=config["sage_layers"],
    linear_channels=config["linear_channels"],
    linear_layers=config["linear_layers"],
)

model.load_state_dict(torch.load(WEIGHTS_PATH))
model = model.to(device)


# |%%--%%| <P7z3qEAf51|Aarqg1Kzan>
from tqdm.auto import tqdm

model.eval()
with torch.no_grad():
    preds_by_file = {}
    for file in tqdm(files):
        data = parse_file(file)
        preds = []
        for d in data:
            d.to(device)
            pred = model(d)
            preds.append(pred.item())

        preds_by_file[file] = np.argsort(preds)[:5]


# |%%--%%| <Aarqg1Kzan|FYjd0c8Z6a>


import pandas as pd

submission = pd.read_csv("data/sample_submission.csv")

preds = pd.DataFrame.from_dict(
    {"file": preds_by_file.keys(), "preds": preds_by_file.values()}
)

preds["ID"] = preds["file"].apply(lambda x: x.split("/")[-1].split(".")[0])
preds["ID"] = "tile:xla:" + preds["ID"]
preds["TopConfigs"] = preds["preds"].apply(lambda x: ";".join([str(i) for i in x]))

preds = preds.drop(columns=["file", "preds"])

joined = submission.merge(preds, on="ID", how="left")

joined["TopConfigs"] = joined["TopConfigs_y"].fillna(joined["TopConfigs_x"])
joined = joined.drop(columns=["TopConfigs_x", "TopConfigs_y"])

joined.to_csv("data/gsage_submission.csv", index=False)


# format file to id
