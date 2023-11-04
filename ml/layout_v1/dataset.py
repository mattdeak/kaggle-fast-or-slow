import hashlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data, Dataset
from tqdm.auto import tqdm

NUM_OPCODES = 121

T = TypeVar("T", torch.Tensor, npt.NDArray[Any])

Transform = Callable[[Any], Any]


class DataPreTransform(Protocol):
    def __call__(
        self,
        x: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """Perform a transform on the data (features or edges or both). Return the transformed data."""
        ...


class ConfigPreTransform(Protocol):
    def __call__(
        self,
        config_features: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        ...


class GlobalPreTransform(Protocol):
    def __call__(
        self,
        x: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        ...


class DataPostTransform(Protocol):
    def __call__(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_config_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a transform on the data (features or edges or both). Return the transformed data."""
        ...


def get_file_id(file_path: str) -> str:
    return os.path.basename(file_path).removesuffix(".npz")


@dataclass
class GraphTensorData:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    node_config_ids: torch.Tensor
    config_features: torch.Tensor
    config_runtime: torch.Tensor
    global_features: torch.Tensor | None = None

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
    GLOBAL_FEATURES_FILE = "global_feat.npy"

    def __init__(
        self,
        *,
        directories: list[str],
        limit: int | None = None,
        max_files_per_config: int | None = None,
        mode: Literal["lazy", "memmapped"] = "memmapped",
        processed_dir: str | None = None,
        force_reload: bool = False,
        data_pre_transform: DataPreTransform | None = None,
        data_post_transform: DataPostTransform | None = None,
        config_pre_transform: ConfigPreTransform | None = None,
        global_pre_transform: GlobalPreTransform | None = None,
        y_transform: Transform | None = None,
        progress: bool = True,
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
        self.force_reload = force_reload

        self.data_pre_transform = data_pre_transform
        self.config_pre_transform = config_pre_transform
        self.global_pre_transform = global_pre_transform
        self.data_post_transform = data_post_transform
        self.y_transform = y_transform
        self.progress = progress

        if mode == "memmapped":
            if processed_dir is None:
                raise ValueError("Must specify processed_dir if mode is processed.")

            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir, exist_ok=True)

            self._processed_dir = processed_dir
            self._processed_file_names = os.listdir(processed_dir)

        # Flattens the files into a list of configs. A single idx corresponds to
        # a file-config pair.
        self.idx_to_source_file_and_config: dict[int, tuple[str, int]] = {}
        self.idx_to_config: dict[int, tuple[str, int]] = {}
        self.idx_groups: list[list[int]] = []
        self._loaded = False
        self._length = 0
        self._groups = 0

        super().__init__()

    def len(self):
        return len(self.idx_to_config)

    def _file_is_processed(self, raw_dir: str, filepath: str) -> bool:
        # Check
        processed_dir = self.get_subdir(raw_dir, filepath)
        return os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0

    def get_subdir(self, raw_dir: str, filepath: str) -> str:
        dir_hash = hashlib.md5(raw_dir.encode("utf-8")).hexdigest()
        processed_subdir = os.path.join(
            self._processed_dir, dir_hash, get_file_id(filepath)
        )
        return processed_subdir

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
        """This function is bad"""
        files = os.listdir(raw_dir)
        for f in tqdm(files, disable=not self.progress):
            filepath = os.path.join(raw_dir, f)

            if self.mode == "memmapped":
                processed_subdir = self.get_subdir(raw_dir, filepath)
                os.makedirs(processed_subdir, exist_ok=True)
            else:
                processed_subdir = None

            d = np.load(filepath)
            num_configs = d["config_runtime"].shape[0]

            if self.max_configs_per_file is not None:
                num_configs = min(self.max_configs_per_file, num_configs)

            if self.mode == "memmapped" and processed_subdir:
                if self.force_reload or not self._file_is_processed(raw_dir, filepath):
                    self.process_to_npy(filepath, processed_subdir)

            if num_configs == 0:
                continue

            self._groups += 1
            self.idx_groups.append(
                list(range(self._length, self._length + num_configs))
            )

            for i in range(num_configs):
                if self.mode == "memmapped" and processed_subdir:
                    self.idx_to_config[self._length] = (processed_subdir, i)
                else:
                    self.idx_to_config[self._length] = (filepath, i)

                self.idx_to_source_file_and_config[self._length] = (filepath, i)
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
                global_features,
            ) = self.extract_from_npy(idx)
        else:
            (
                node_features,
                edge_index,
                node_config_ids,
                config_features,
                config_runtime,
                global_features,
            ) = self.extract_from_npz(idx)

        processed_config_features = torch.zeros(
            (node_features.shape[0], config_features.shape[1])
        )
        processed_config_features[node_config_ids] = config_features
        all_features = torch.cat([node_features, processed_config_features], axis=1)
        all_features = cast(torch.Tensor, all_features)

        all_features, edge_index = (
            self.data_post_transform(all_features, edge_index, node_config_ids)
            if self.data_post_transform
            else (all_features, edge_index)
        )
        y = self.y_transform(config_runtime) if self.y_transform else config_runtime

        return Data(
            x=all_features,  # type: ignore
            edge_index=edge_index.T.contiguous(),
            y=y,
        )

    def process_to_npy(self, source_file_path: str, target_file_path: str) -> None:
        d = np.load(source_file_path)
        node_features = d["node_feat"]
        node_opcodes = d["node_opcode"]
        edge_index = d["edge_index"]
        node_config_feat = d["node_config_feat"]
        node_config_ids = d["node_config_ids"]
        config_runtime = d["config_runtime"]

        ohe_opcodes = np.zeros((node_opcodes.shape[0], NUM_OPCODES))
        ohe_opcodes[np.arange(node_opcodes.shape[0]), node_opcodes] = 1
        node_features = np.concatenate([node_features, ohe_opcodes], axis=1)

        if self.data_pre_transform:
            node_features, edge_index, node_config_ids = self.data_pre_transform(
                node_features, edge_index, node_config_ids
            )

        if self.config_pre_transform:
            node_config_feat = self.config_pre_transform(node_config_feat)

        assert (
            config_runtime.shape[0] == node_config_feat.shape[0]
        ), f"Config shape mismatch on source file: {source_file_path}"

        np.save(
            os.path.join(target_file_path, self.NODE_FEATURES_FILE),
            node_features.astype(np.float32),
        )
        np.save(
            os.path.join(target_file_path, self.EDGE_INDEX_FILE),
            edge_index.astype(np.int64),
        )
        np.save(
            os.path.join(target_file_path, self.NODE_IDS_FILE),
            node_config_ids.astype(np.int64),
        )
        np.save(
            os.path.join(target_file_path, self.CONFIG_FEATURES_FILE),
            node_config_feat.astype(np.float32),
        )
        np.save(
            os.path.join(target_file_path, self.CONFIG_RUNTIME_FILE),
            config_runtime.astype(np.int64),
        )
        if self.global_pre_transform:
            global_features = self.global_pre_transform(
                node_features, edge_index, node_config_ids
            )
            np.save(
                os.path.join(target_file_path, self.GLOBAL_FEATURES_FILE),
                global_features.astype(np.float32),
            )

    def extract_from_npz(self, idx: int) -> GraphTensorData:
        file_path, config_idx = self.idx_to_config[idx]
        d = np.load(file_path)
        node_features = d["node_feat"]
        opcodes = d["node_opcode"]
        edge_index = d["edge_index"]
        node_config_feat = d["node_config_feat"][config_idx, :, :]
        node_config_ids = d["node_config_ids"]
        config_runtime = d["config_runtime"][config_idx]

        # One hot encode the opcodes
        ohe_opcodes = np.zeros((opcodes.shape[0], NUM_OPCODES))
        ohe_opcodes[np.arange(opcodes.shape[0]), opcodes] = 1
        node_features = np.concatenate([node_features, ohe_opcodes], axis=1)

        return GraphTensorData(
            node_features=torch.from_numpy(node_features).float(),
            edge_index=torch.from_numpy(edge_index),
            node_config_ids=torch.from_numpy(node_config_ids),
            config_features=torch.from_numpy(node_config_feat).float(),
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

        # debug
        config_features = torch.from_numpy(
            np.array(
                np.load(
                    os.path.join(file_path, self.CONFIG_FEATURES_FILE),
                    mmap_mode="r+",
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

        global_features = None
        if os.path.exists(os.path.join(file_path, self.GLOBAL_FEATURES_FILE)):
            global_features = torch.from_numpy(
                np.load(
                    os.path.join(file_path, self.GLOBAL_FEATURES_FILE),
                    mmap_mode="r",
                )
            )

        return GraphTensorData(
            node_features=node_features,
            edge_index=edge_index,
            node_config_ids=node_config_ids,
            config_features=config_features,
            config_runtime=config_runtime,
            global_features=global_features,
        )


class ConcatenatedDataset(Dataset):
    def __init__(self, dataset1: LayoutDataset, dataset2: LayoutDataset):
        self.datasets = [dataset1, dataset2]
        self.idx_groups = self.get_idx_groups()

    def get_idx_groups(self):
        ds1_groups = self.datasets[0].idx_groups
        ds2_groups = self.datasets[1].idx_groups

        ds2_groups = [[x + len(self.datasets[0]) for x in g] for g in ds2_groups]
        return ds1_groups + ds2_groups

    def len(self):
        return len(self.datasets[0]) + len(self.datasets[1])

    def get(self, idx: int) -> Data:
        if idx < len(self.datasets[0]):
            return self.datasets[0].get(idx)
        else:
            return self.datasets[1].get(idx - len(self.datasets[0]))
