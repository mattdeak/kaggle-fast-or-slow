import os
from typing import Any, Callable, Literal, TypeVar

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

T = TypeVar("T")

Transform = Callable[[Any], Any]


class LayoutDataset(Dataset):
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
        directories: list[str],
        limit: int | None = None,
        max_files_per_config: int | None = None,
        mode: Literal["lazy", "processed", "cached"] = "lazy",
        processed_dir: str | None = None,
    ):
        """Directories should be a list of directories to load from.

        If limit is not None, then only the first limit files will be loaded.
        If cache_graphs, then the all graph features except configs will be
        cached in memory. This is better if you can fit it in memory, as the
        reads from disk are slow."""
        self.directories = directories
        self._processed_file_names: list[str] = []
        self.max_configs_per_file = max_files_per_config
        self.limit = limit
        self.mode = mode

        if mode == "processed":
            if processed_dir is None:
                raise ValueError("Must specify processed_dir if mode is processed.")
            self._processed_dir = processed_dir
            self._processed_file_names = os.listdir(processed_dir)

        self.lookup_table = {}
        # Flattens the files into a list of configs. A single idx corresponds to
        # a file-config pair.
        self.idx_to_config: dict[int, tuple[str, int]] = {}
        self.graph_cache: dict[int, Data] = {}
        self._loaded = False
        self._length = 0

        super().__init__()

    def len(self):
        return len(self.idx_to_config)

    def load(self):
        # We don't do anything except load the files into our lookup
        # table
        for raw_dir in self.directories:
            self._load_dir(raw_dir)

    def _load_dir(self, raw_dir: str):
        files = os.listdir(raw_dir)
        for f in files:
            filepath = os.path.join(raw_dir, f)
            d = np.load(filepath)
            num_configs = d["config_runtime"].shape[0]
            if self.max_configs_per_file is not None:
                num_configs = min(self.max_configs_per_file, num_configs)

            for i in range(num_configs):
                self.idx_to_config[self._length] = (filepath, i)
                if self.mode == "cached":
                    self.graph_cache[self._length] = self.process_from_npz(self._length)
                elif self.mode == "processed":
                    data = self.process_from_npz(self._length)
                    torch.save(data, self._processed_dir + f"_{i}.pt")

                self._length += 1

        self._loaded = True

    def get(self, idx: int) -> Data:
        if self.mode == "cached":
            raise NotImplementedError
        elif self.mode == "processed":
            return torch.load(self._processed_dir + f"_{idx}.pt")
        else:
            return self.process_from_npz(idx)

    def process_from_npz(self, idx: int) -> Data:
        file_path, config_idx = self.idx_to_config[idx]
        d = np.load(file_path)
        edge_index = d["edge_index"]
        node_features = d["node_feat"]

        node_opcode = d["node_opcode"]
        ohe_opcodes = np.zeros((node_opcode.shape[0], 121))
        ohe_opcodes[np.arange(node_opcode.shape[0]), node_opcode] = 1

        node_config_feat = d["node_config_feat"]
        node_config_ids = d["node_config_ids"]
        config_runtime = d["config_runtime"]

        config_features_raw = node_config_feat[config_idx, :, :]
        config_features = np.zeros(
            (node_features.shape[0], config_features_raw.shape[1])
        )
        config_features[node_config_ids] = config_features_raw
        all_features = np.concatenate(
            (node_features, ohe_opcodes, config_features), axis=1
        )

        return Data(
            x=torch.from_numpy(all_features).float(),  # type: ignore
            edge_index=torch.from_numpy(edge_index.T),  # type: ignore
            y=torch.log(torch.tensor(config_runtime[config_idx])),  # type: ignore
        )
