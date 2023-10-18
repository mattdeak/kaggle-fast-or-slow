import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data, Dataset

T = TypeVar("T")

Transform = Callable[[Any], Any]


def get_file_id(file_path: str) -> str:
    return os.path.basename(file_path).split(".")[0]


@dataclass
class GraphTensorData:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    node_config_ids: torch.Tensor
    config_features: torch.Tensor
    config_runtime: torch.Tensor

    def __iter__(self):
        return iter(
            (
                self.node_features,
                self.edge_index,
                self.node_config_ids,
                self.config_features,
                self.config_runtime,
            )
        )


class LayoutDataset(Dataset):
    """I actually hate the base class for this, but
    the dataloader that requires this has some nice utilities for
    batching graphs."""

    FILES_TO_IGNORE = [
        "pre_filter.pt",
        "pre_transform.pt",
    ]

    NODE_FEATURES_FILE = "node_feat.npy"
    EDGE_INDEX_FILE = "edge_index.npy"
    CONFIG_FEATURES_FILE = "node_config_feat.npy"
    NODE_IDS_FILE = "node_config_ids.npy"
    CONFIG_RUNTIME_FILE = "config_runtime.npy"

    def __init__(
        self,
        *,
        directories: list[str],
        limit: int | None = None,
        max_files_per_config: int | None = None,
        mode: Literal["lazy", "memmapped"] = "memmapped",
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

        if mode == "memmapped":
            if processed_dir is None:
                raise ValueError("Must specify processed_dir if mode is processed.")

            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir, exist_ok=True)

            self._processed_dir = processed_dir
            self._processed_file_names = os.listdir(processed_dir)

        # Flattens the files into a list of configs. A single idx corresponds to
        # a file-config pair.
        self.idx_to_config: dict[int, tuple[str, int]] = {}
        self.idx_to_processed_dir: dict[int, str] = {}
        self._loaded = False
        self._length = 0

        super().__init__()

    def len(self):
        return len(self.idx_to_config)

    def load(self):
        # We don't do anything except load the files into our lookup
        # table
        for raw_dir in self.directories:
            if self.mode == "memmapped":
                os.makedirs(
                    os.path.join(self._processed_dir, get_file_id(raw_dir)),
                    exist_ok=True,
                )

            self._load_dir(raw_dir)

    def _load_dir(self, raw_dir: str):
        """This function is shit"""
        files = os.listdir(raw_dir)
        for f in files:
            filepath = os.path.join(raw_dir, f)

            if self.mode == "memmapped":
                processed_subdir = os.path.join(
                    self._processed_dir, get_file_id(raw_dir), get_file_id(filepath)
                )
                os.makedirs(processed_subdir, exist_ok=True)
            else:
                processed_subdir = None

            d = np.load(filepath)
            num_configs = d["config_runtime"].shape[0]
            if self.max_configs_per_file is not None:
                num_configs = min(self.max_configs_per_file, num_configs)

            if self.mode == "memmapped" and processed_subdir:
                self.process_to_npy(filepath, processed_subdir)

            for i in range(num_configs):
                if self.mode == "memmapped" and processed_subdir:
                    self.idx_to_config[self._length] = (processed_subdir, i)
                else:
                    self.idx_to_config[self._length] = (filepath, i)

                self._length += 1

        self._loaded = True

    def get(self, idx: int) -> Data:
        if self.mode == "memmapped":
            (
                node_features,
                edge_index,
                node_config_ids,
                config_features,
                config_runtime,
            ) = self.extract_from_npy(idx)
        else:
            (
                node_features,
                edge_index,
                node_config_ids,
                config_features,
                config_runtime,
            ) = self.extract_from_npz(idx)

        processed_config_features = torch.zeros(
            (node_features.shape[0], config_features.shape[1])
        )
        processed_config_features[node_config_ids] = config_features
        all_features = torch.cat([node_features, processed_config_features], axis=1)

        return Data(
            x=all_features,  # type: ignore
            edge_index=edge_index,
            y=torch.log(config_runtime),
        )

    def process_to_npy(self, source_file_path: str, target_file_path: str) -> None:
        d = np.load(source_file_path)
        node_features = d["node_feat"]
        edge_index = d["edge_index"]
        node_config_feat = d["node_config_feat"]
        node_config_ids = d["node_config_ids"]
        config_runtime = d["config_runtime"]

        np.save(
            os.path.join(target_file_path, self.NODE_FEATURES_FILE),
            node_features,
        )
        np.save(
            os.path.join(target_file_path, self.EDGE_INDEX_FILE),
            edge_index,
        )
        np.save(
            os.path.join(target_file_path, self.NODE_IDS_FILE),
            node_config_ids,
        )
        np.save(
            os.path.join(target_file_path, self.CONFIG_FEATURES_FILE),
            node_config_feat,
        )
        np.save(
            os.path.join(target_file_path, self.CONFIG_RUNTIME_FILE),
            config_runtime,
        )

    def extract_from_npz(self, idx: int) -> GraphTensorData:
        file_path, config_idx = self.idx_to_config[idx]
        d = np.load(file_path)
        node_features = d["node_feat"]
        edge_index = d["edge_index"]
        node_config_feat = d["node_config_feat"][config_idx, :, :]
        node_config_ids = d["node_config_ids"]
        config_runtime = float(d["config_runtime"][config_idx])

        return GraphTensorData(
            node_features=torch.from_numpy(node_features),
            edge_index=torch.from_numpy(edge_index),
            node_config_ids=torch.from_numpy(node_config_ids),
            config_features=torch.from_numpy(node_config_feat),
            config_runtime=torch.tensor(config_runtime),
        )

    def extract_from_npy(self, idx: int) -> GraphTensorData:
        file_path, config_idx = self.idx_to_config[idx]
        node_features = torch.from_numpy(
            np.load(
                os.path.join(file_path, self.NODE_FEATURES_FILE),
            )
        )
        edge_index = torch.from_numpy(
            np.load(
                os.path.join(file_path, self.EDGE_INDEX_FILE),
            )
        )

        node_config_ids = torch.from_numpy(
            np.load(
                os.path.join(file_path, self.NODE_IDS_FILE),
            )
        )

        config_features = torch.from_numpy(
            np.array(
                np.load(
                    os.path.join(file_path, self.CONFIG_FEATURES_FILE),
                    mmap_mode="r",
                )[config_idx, :, :]
            )
        )
        config_runtime = torch.tensor(
            float(
                np.load(
                    os.path.join(file_path, self.CONFIG_RUNTIME_FILE),
                    mmap_mode="r",
                )[config_idx]
            )
        )

        return GraphTensorData(
            node_features=node_features,
            edge_index=edge_index,
            node_config_ids=node_config_ids,
            config_features=config_features,
            config_runtime=config_runtime,
        )
