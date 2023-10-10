import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm.auto import tqdm

from ml.xla_gcn_v1.dataset import XLATileDataset, parse_file

# |%%--%%| <Qm2sDuudhp|I0yQRtnqYq>


class ModifiedGCN(torch.nn.Module):
    def __init__(
        self,
        graph_input_dim: int,
        global_input_dim: int,
        gcn_out_dims: list[int],
        linear_dims: list[int],
        output_dim: int,
    ):
        super().__init__()

        initial_conv = GCNConv(graph_input_dim, gcn_out_dims[0])
        self.convs = torch.nn.ModuleList(
            [initial_conv]
            + [GCNConv(i, o) for i, o in zip(gcn_out_dims[:-1], gcn_out_dims[1:])]
        )

        first_linear = torch.nn.Linear(
            gcn_out_dims[-1] + global_input_dim, linear_dims[0]
        )

        self.fcs = torch.nn.ModuleList(
            [first_linear]
            + [torch.nn.Linear(i, o) for i, o in zip(linear_dims[:-1], linear_dims[1:])]
        )

        output_linear = torch.nn.Linear(linear_dims[-1], output_dim)
        self.fcs.append(output_linear)

    def forward(self, data: Data):
        x, edge_index, global_features = (
            data.x,
            data.edge_index,
            data["global_features"],
        )

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.leaky_relu(x)

        pool = global_mean_pool(x, data.batch)
        x = torch.cat((pool, global_features), dim=1)

        for fc in self.fcs:
            x = fc(x)
            x = F.leaky_relu(x)

        return x


# |%%--%%| <I0yQRtnqYq|NhXS6Kuh4Q>

INPUT_DIM = 261
GLOBAL_INPUT_DIM = 24

GCN_DIMS = [64, 32, 16]
LINEAR_DIMS = [128, 64, 32]

nn = ModifiedGCN(INPUT_DIM, GLOBAL_INPUT_DIM, GCN_DIMS, LINEAR_DIMS, 1)

# |%%--%%| <NhXS6Kuh4Q|1NKjfOoHTI>

TRAIN_DIR = "data/npz/tile/xla/train"
VALID_DIR = "data/npz/tile/xla/valid"
train_dataset = XLATileDataset(
    processed="data/processed/train", raw=TRAIN_DIR, max_files_per_config=1000
)

valid_dataset = XLATileDataset(processed="data/processed/valid", raw=VALID_DIR)


# |%%--%%| <1NKjfOoHTI|w6oI8NpWeo>


train_loader = DataLoader(train_dataset, batch_size=32)
valid_loader = DataLoader(valid_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

import wandb

model.train()


LOG_INTERVAL = 100
EVAL_INTERVAL = 1000

with wandb.init(project="kaggle-fast-or-slow"):
    wandb.watch(model)
    model.train()
    for i, batch in tqdm(enumerate(train_loader)):
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out.flatten(), batch.y)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            wandb.log({"train_loss": loss.item()})

        if i % EVAL_INTERVAL == 0:
            model.eval()
            validation_loss = 0
            with torch.no_grad():
                for batch in tqdm(valid_loader):
                    batch.to(device)
                    out = model(batch)
                    loss = F.mse_loss(out.flatten(), batch.y)
                    validation_loss += loss.item()

            validation_loss /= len(valid_loader)
            wandb.log({"valid_loss": validation_loss})
            model.train()
