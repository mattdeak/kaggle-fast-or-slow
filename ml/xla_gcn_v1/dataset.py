import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


def parse_file(file_path: str) -> list[Data]:
    """parses a tile xla npz file into a list torch geometric data object."""
    data = dict(np.load(file_path))

    ohe_opcodes = np.zeros((data["node_opcode"].shape[0], 121))
    ohe_opcodes[np.arange(data["node_opcode"].shape[0]), data["node_opcode"]] = 1
    node_features = torch.tensor(
        np.hstack((data["node_feat"], ohe_opcodes)), dtype=torch.float
    )
    edge_index = torch.tensor(data["edge_index"].T)
    target = torch.tensor(
        np.log(data["config_runtime"] / data["config_runtime_normalizers"]),
        dtype=torch.float,
    )

    return [
        Data(
            x=node_features,
            edge_index=edge_index,
            y=target[i],
            global_features=torch.tensor(data["config_feat"][i]).reshape(1, -1),
        )
        for i in range(data["config_feat"].shape[0])
    ]


class XLATileDataset(Dataset):
    """I actually hate the base class for this, but
    the dataloader that requires this has some nice utilities for
    batching graphs."""

    FILES_TO_IGNORE = [
        "pre_filter.pt",
        "pre_transform.pt",
    ]

    def __init__(self, *, processed: str, raw: str):
        self.processed = processed
        self.raw = raw
        self._processed_file_names: list[str] = []
        super().__init__()

    @property
    def raw_dir(self):
        return self.raw

    @property
    def processed_dir(self):
        return self.processed

    @property
    def raw_file_names(self):
        return os.listdir(self.raw)

    @property
    def processed_file_names(self) -> list[str]:
        return self._processed_file_names

    def process(self):
        # We're just going to manually check if
        # the processed data exists or not
        # because the default way is  dumb

        processed_files: list[str] = glob.glob(os.path.join(self.processed_dir, "*.pt"))
        processed_files = [
            file for file in processed_files if file not in self.FILES_TO_IGNORE
        ]

        if len(processed_files) > 0:
            self._processed_file_names = os.listdir(self.processed_dir)
            return

        with ThreadPoolExecutor(max_workers=18) as pool:
            args = [
                (raw_path, identifier)
                for identifier, raw_path in enumerate(self.raw_paths)
            ]
            results = pool.map(
                self._process_file,
                (raw_path for raw_path, _ in args),
                (identifier for _, identifier in args),
            )

        print(results)

        for result in results:
            print(result)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx: int):
        data = torch.load(
            os.path.join(self.processed_dir, self._processed_file_names[idx])
        )
        return data

    def _process_file(self, raw_path: str, identifier: int) -> None:
        print("Processing", raw_path)
        all_data = parse_file(raw_path)

        for jdx, d in enumerate(all_data):
            filename = f"data_{identifier}_config{jdx}.pt"
            torch.save(d, os.path.join(self.processed_dir, filename))
            self._processed_file_names.append(filename)
