import glob
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm.contrib.concurrent import thread_map


def parse_file(file_path: str) -> list[Data]:
    """parses a tile xla npz file into a list torch geometric data object."""
    data = dict(np.load(file_path))

    ohe_opcodes = np.zeros((data["node_opcode"].shape[0], 121))
    ohe_opcodes[np.arange(data["node_opcode"].shape[0]), data["node_opcode"]] = 1
    node_features = torch.tensor(
        np.hstack((data["node_feat"], ohe_opcodes)), dtype=torch.float32
    )
    edge_index = torch.tensor(data["edge_index"].T)
    target = torch.tensor(
        np.log(data["config_runtime"] / data["config_runtime_normalizers"]),
        dtype=torch.float32,
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

    def __init__(
        self,
        *,
        processed: str,
        raw: str,
        limit: int | None = None,
        max_files_per_config: int | None = None,
        shuffle_configs: bool = True,
    ):
        self.processed = processed
        self.raw = raw
        self._processed_file_names: list[str] = []
        self.max_files_per_config = max_files_per_config
        self.limit = limit
        self.shuffle_configs = shuffle_configs
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

            # Remove the files to ignore
            self._processed_file_names = [
                file
                for file in self._processed_file_names
                if file not in self.FILES_TO_IGNORE
            ]
            return

        if self.limit is not None:
            raw_paths = self.raw_paths[: self.limit]
        else:
            raw_paths = self.raw_paths

        args = [(raw_path, identifier) for identifier, raw_path in enumerate(raw_paths)]
        thread_map(
            self._process_file,
            (raw_path for raw_path, _ in args),
            (identifier for _, identifier in args),
            max_workers=16,
            total=len(args),
        )

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx: int):
        data = torch.load(
            os.path.join(self.processed_dir, self._processed_file_names[idx])
        )
        return data

    def _process_file(self, raw_path: str, identifier: int) -> None:
        all_data = parse_file(raw_path)

        # randomly shuffle
        if self.shuffle_configs:
            random.shuffle(all_data)

        for jdx, d in enumerate(all_data):
            filename = f"data_{identifier}_config{jdx}.pt"
            torch.save(d, os.path.join(self.processed_dir, filename))
            self._processed_file_names.append(filename)

            if self.max_files_per_config and jdx >= self.max_files_per_config:
                break
