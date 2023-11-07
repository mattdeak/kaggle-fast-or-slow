import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data, Dataset
from tqdm.auto import tqdm
from tqdm.contrib import concurrent
from tqdm.contrib.concurrent import process_map

from ml.layout_v1.preprocessors import ohe_opcodes


class DataTransform(Protocol):
    def __call__(
        self,
        x: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """Perform a transform on the data (features or edges or both). Return the transformed data."""
        ...


class GraphTransform(Protocol):
    def __call__(
        self,
        node_features: npt.NDArray[Any],
        opcodes: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """Perform a transform on the data (features or edges or both). Return the transformed data."""
        ...


class OpcodeEmbedder(Protocol):
    def __call__(
        self,
        opcodes: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Perform a transform the opcodes to generate an embedding."""
        ...


class ConfigTransform(Protocol):
    def __call__(
        self,
        config_features: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        ...


class GlobalTransform(Protocol):
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


class TargetTransform(Protocol):
    def __call__(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Perform a transform on the target. Return the transformed target."""
        ...


def get_file_id(file_path: str) -> str:
    return os.path.basename(file_path).removesuffix(".npz")


@dataclass
class GraphNumpyData:
    node_features: npt.NDArray[Any]
    opcode_embeds: npt.NDArray[Any]
    edge_index: npt.NDArray[Any]
    node_config_ids: npt.NDArray[Any]
    config_features: npt.NDArray[Any]
    config_runtime: npt.NDArray[Any]
    global_features: npt.NDArray[Any] | None = None

    def __iter__(self):
        return iter(
            (
                self.node_features,
                self.opcode_embeds,
                self.edge_index,
                self.node_config_ids,
                self.config_features,
                self.config_runtime,
                self.global_features,
            )
        )


@dataclass
class Transforms:
    node_transform: DataTransform | None = None
    opcode_transform: OpcodeEmbedder | None = None
    graph_transform: GraphTransform | None = None
    config_transform: ConfigTransform | None = None
    global_transform: GlobalTransform | None = None
    target_transform: TargetTransform | None = None


@dataclass
class ProcessResult:
    num_configs: int
    idx_to_config: dict[int, tuple[str, int]]
    idx_to_source_file_and_config: dict[int, tuple[str, int]]


class LayoutDataset(Dataset):
    """I actually hate the base class for this, but
    the dataloader that requires this has some nice utilities for
    batching graphs."""

    FILES_TO_IGNORE = [
        "pre_filter.pt",
        "pre_transform.pt",
    ]

    NODE_FEATURES_FILE = "node_feat.npy"
    OPCODE_EMBEDDINGS_FILE = "node_opcode.npy"
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
        node_pre_transform: DataTransform | None = None,
        node_post_transform: DataTransform | None = None,
        opcode_pre_transform: OpcodeEmbedder | None = None,
        opcode_post_transform: OpcodeEmbedder | None = None,
        graph_pre_transform: GraphTransform | None = None,
        graph_post_transform: GraphTransform | None = None,
        config_pre_transform: ConfigTransform | None = None,
        config_post_transform: ConfigTransform | None = None,
        global_pre_transform: GlobalTransform | None = None,
        global_post_transform: GlobalTransform | None = None,
        target_pre_transform: TargetTransform | None = None,
        target_post_transform: TargetTransform | None = None,
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

        self.pretransforms = Transforms(
            node_transform=node_pre_transform,
            opcode_transform=opcode_pre_transform,
            graph_transform=graph_pre_transform,
            config_transform=config_pre_transform,
            global_transform=global_pre_transform,
            target_transform=target_pre_transform,
        )

        self.posttransforms = Transforms(
            node_transform=node_post_transform,
            opcode_transform=opcode_post_transform,
            graph_transform=graph_post_transform,
            config_transform=config_post_transform,
            global_transform=global_post_transform,
            target_transform=target_post_transform,
        )

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
        with ProcessPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(
                        self.process_file,
                        [raw_dir] * len(files),
                        [os.path.join(raw_dir, f) for f in files],
                    ),
                    total=len(files),
                    disable=not self.progress,
                )
            )

        for result in results:
            if result.num_configs == 0:
                continue

            self._groups += 1
            self.idx_groups.append(
                list(range(self._length, self._length + result.num_configs))
            )

            for idx, config in result.idx_to_config.items():
                self.idx_to_config[self._length] = config
                self.idx_to_source_file_and_config[
                    self._length
                ] = result.idx_to_source_file_and_config[idx]
                self._length += 1

        self._loaded = True

    def process_file(self, raw_dir: str, file_path: str) -> ProcessResult:
        """Process a single file. This will save the processed file to disk.,
        And then return the number of configs in the file that were processed."""

        if self.mode == "memmapped":
            processed_subdir = self.get_subdir(raw_dir, file_path)
            os.makedirs(processed_subdir, exist_ok=True)
        else:
            processed_subdir = None

        d = np.load(file_path)
        num_configs = d["config_runtime"].shape[0]

        if self.max_configs_per_file is not None:
            num_configs = min(self.max_configs_per_file, num_configs)

        if self.mode == "memmapped" and processed_subdir:
            if self.force_reload or not self._file_is_processed(raw_dir, file_path):
                self.process_to_npy(file_path, processed_subdir)

        if num_configs == 0:
            return ProcessResult(0, {}, {})

        # number of groups
        idx_to_config = {}
        idx_to_source_file_and_config = {}

        for i in range(num_configs):
            if self.mode == "memmapped" and processed_subdir:
                idx_to_config[i] = (processed_subdir, i)
            else:
                idx_to_config[i] = (file_path, i)

            idx_to_source_file_and_config[i] = (file_path, i)

        return ProcessResult(num_configs, idx_to_config, idx_to_source_file_and_config)

    def get(self, idx: int) -> Data:
        if self.mode == "memmapped":
            (
                node_features,
                opcode_embeds,
                edge_index,
                node_config_ids,
                config_features,
                config_runtime,
                global_features,
            ) = self.extract_from_npy(idx)
        else:
            (
                node_features,
                opcode_embeds,
                edge_index,
                node_config_ids,
                config_features,
                config_runtime,
                global_features,
            ) = self.extract_from_npz(idx)

        if config_features.ndim == 2:
            config_features = config_features[np.newaxis, :, :]

        (
            node_features,
            opcode_embeds,
            edge_index,
            node_config_ids,
            config_features,
            config_runtime,
            global_features,
        ) = self.apply_transforms(
            node_features=node_features,
            opcodes=opcode_embeds,
            edge_index=edge_index,
            node_config_ids=node_config_ids,
            config_features=config_features,
            config_runtime=config_runtime,
            transforms=self.posttransforms,
        )

        processed_config_features = np.zeros(
            (node_features.shape[0], config_features.shape[-1])
        )
        processed_config_features[node_config_ids] = config_features

        all_features = np.concatenate(
            [node_features, opcode_embeds, processed_config_features], axis=1
        )

        return Data(
            x=torch.from_numpy(all_features),  # type: ignore
            edge_index=torch.from_numpy(edge_index).T.contiguous(),
            y=torch.from_numpy(config_runtime),
            global_features=torch.from_numpy(global_features)
            if global_features
            else None,
        )

    def process_to_npy(self, source_file_path: str, target_file_path: str) -> None:
        d = np.load(source_file_path)
        node_features: npt.NDArray[Any] = d["node_feat"]
        node_opcodes: npt.NDArray[Any] = d["node_opcode"]
        edge_index: npt.NDArray[Any] = d["edge_index"]
        node_config_feat: npt.NDArray[Any] = d["node_config_feat"]
        node_config_ids: npt.NDArray[Any] = d["node_config_ids"]
        config_runtime: npt.NDArray[Any] = d["config_runtime"]

        (
            node_features,
            node_opcodes,
            edge_index,
            node_config_ids,
            node_config_feat,
            config_runtime,
            global_features,
        ) = self.apply_transforms(
            node_features=node_features,
            opcodes=node_opcodes,
            edge_index=edge_index,
            node_config_ids=node_config_ids,
            config_features=node_config_feat,
            config_runtime=config_runtime,
            transforms=self.pretransforms,
        )

        assert (
            config_runtime.shape[0] == node_config_feat.shape[0]
        ), f"Config shape mismatch on source file: {source_file_path}"

        np.save(
            os.path.join(target_file_path, self.NODE_FEATURES_FILE),
            node_features.astype(np.float32),
        )

        np.save(
            os.path.join(target_file_path, self.OPCODE_EMBEDDINGS_FILE),
            node_opcodes.astype(np.float32),
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
        if global_features:
            np.save(
                os.path.join(target_file_path, self.GLOBAL_FEATURES_FILE),
                global_features.astype(np.float32),
            )

    def extract_from_npz(self, idx: int) -> GraphNumpyData:
        file_path, config_idx = self.idx_to_config[idx]
        d = np.load(file_path)
        node_features = d["node_feat"]
        opcodes = d["node_opcode"]
        edge_index = d["edge_index"]
        node_config_feat = d["node_config_feat"][config_idx, :, :]
        node_config_ids = d["node_config_ids"]
        config_runtime = d["config_runtime"][config_idx]

        return GraphNumpyData(
            node_features=node_features,
            opcode_embeds=opcodes,
            edge_index=edge_index,
            node_config_ids=node_config_ids,
            config_features=node_config_feat,
            config_runtime=config_runtime,
        )

    def extract_from_npy(self, idx: int) -> GraphNumpyData:
        file_path, config_idx = self.idx_to_config[idx]
        node_features = np.load(
            os.path.join(file_path, self.NODE_FEATURES_FILE),
        )

        opcodes = np.load(
            os.path.join(file_path, self.OPCODE_EMBEDDINGS_FILE),
        )

        edge_index = np.load(
            os.path.join(file_path, self.EDGE_INDEX_FILE),
        )

        node_config_ids = np.load(
            os.path.join(file_path, self.NODE_IDS_FILE),
        )

        # debug
        config_features = np.array(
            np.load(
                os.path.join(file_path, self.CONFIG_FEATURES_FILE),
                mmap_mode="r+",
            )[config_idx, :, :]
        )

        config_runtime = np.load(
            os.path.join(file_path, self.CONFIG_RUNTIME_FILE),
            mmap_mode="r",
        )[config_idx]

        global_features = None
        if os.path.exists(os.path.join(file_path, self.GLOBAL_FEATURES_FILE)):
            global_features = np.load(
                os.path.join(file_path, self.GLOBAL_FEATURES_FILE),
                mmap_mode="r",
            )

        return GraphNumpyData(
            node_features=node_features,
            opcode_embeds=opcodes,
            edge_index=edge_index,
            node_config_ids=node_config_ids,
            config_features=config_features,
            config_runtime=config_runtime,
            global_features=global_features,
        )

    def apply_transforms(
        self,
        *,
        node_features: npt.NDArray[Any],
        opcodes: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
        config_features: npt.NDArray[Any],
        config_runtime: npt.NDArray[Any],
        transforms: Transforms,
    ) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any] | None,
    ]:
        if transforms.graph_transform:
            (
                node_features,
                opcodes,
                edge_index,
                node_config_ids,
            ) = transforms.graph_transform(
                node_features, opcodes, edge_index, node_config_ids
            )

        if transforms.node_transform:
            node_features, edge_index, node_config_ids = transforms.node_transform(
                node_features, edge_index, node_config_ids
            )

        if transforms.opcode_transform:
            opcodes = transforms.opcode_transform(opcodes)

        if transforms.config_transform:
            config_features = transforms.config_transform(config_features)

        global_features = None
        if transforms.global_transform:
            global_features = transforms.global_transform(
                node_features, edge_index, node_config_ids
            )

        if transforms.target_transform:
            config_runtime = transforms.target_transform(config_runtime)

        return (
            node_features,
            opcodes,
            edge_index,
            node_config_ids,
            config_features,
            config_runtime,
            global_features,
        )


class ConcatenatedDataset(Dataset):
    def __init__(self, dataset1: LayoutDataset, dataset2: LayoutDataset):
        self.datasets = [dataset1, dataset2]
        self.idx_groups = self.get_idx_groups()

        super().__init__()

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
