from typing import Any, Literal, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import torch

from ml.layout_v1.stats import (NLP_TRAIN_NODE_MEANS, NLP_TRAIN_NODE_STDEVS,
                                NUMERIC_FEATURE_MASK, XLA_TRAIN_NODE_MEANS,
                                XLA_TRAIN_NODE_STDEVS)

NUM_OPCODES = 121


def ohe_opcodes(opcodes: npt.NDArray[Any]) -> npt.NDArray[Any]:
    ohe_opcodes = np.zeros((opcodes.shape[0], NUM_OPCODES))
    ohe_opcodes[np.arange(opcodes.shape[0]), opcodes] = 1
    return ohe_opcodes


class OpcodeGroupOHEEmbedder:
    # See the notebook "opcode analysis" for how these groups were determined
    GROUPS = {
        0: [2, 95, 59, 32, 60, 1, 66, 73, 50, 51, 37, 38, 94, 81, 103, 90],
        1: [7, 30, 91, 96, 15, 40, 80, 118, 107],
        2: [53, 52, 54, 55, 87, 88, 89, 11, 12, 65, 19],
        3: [20, 83, 17, 49],
        4: [34, 26, 16, 99],
        5: [27, 29, 28, 35, 36, 42, 82, 22, 92, 108, 75, 76, 98, 62, 93],
        6: [48, 77, 79, 78, 13, 21, 67, 46, 25, 12, 119],
        7: [70, 72, 71, 111, 5, 109, 110, 10, 9, 8, 57, 58, 104, 112, 113],
        8: [14, 23, 102],
        9: [
            85,
            86,
            68,
            69,
            47,
            61,
            6,
            18,
            105,
            106,
            104,
            112,
            113,
            5,
            109,
            110,
            111,
            84,
        ],
        10: [4, 100, 45, 44, 43, 63, 64, 74, 33],
        11: [115, 116, 117, 114],
        12: [63, 24],
        13: [31],
        14: [41],
    }

    def __call__(self, opcodes: npt.NDArray[Any]) -> npt.NDArray[Any]:
        group_opcodes = np.zeros((opcodes.shape[0], len(self.GROUPS)))
        for group_num, group in self.GROUPS.items():
            group_opcodes[np.isin(opcodes, group), group_num] = 1
        return group_opcodes


def reduce_to_config_node_communities_ndarray(
    node_features: npt.NDArray[Any],
    opcodes: npt.NDArray[Any],
    edge_index: npt.NDArray[Any],
    node_config_ids: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    rows, _ = np.where(np.isin(edge_index, node_config_ids))

    new_edge_index = edge_index[rows]
    new_node_config_ids = node_config_ids.copy()
    kept_nodes = np.unique(new_edge_index)

    for i, n in enumerate(kept_nodes):
        new_edge_index[new_edge_index == n] = i
        new_node_config_ids[new_node_config_ids == n] = i

    node_features = node_features[kept_nodes, :]
    opcodes = opcodes[kept_nodes]

    return node_features, opcodes, new_edge_index, new_node_config_ids


def drop_features_xla(
    x: npt.NDArray[Any], edge_index: npt.NDArray[Any], node_config_ids: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    x = np.delete(x, XLA_DROP_FEATURES, axis=1)
    return x, edge_index, node_config_ids


def log_transform_specific_features(
    x: npt.NDArray[Any], edge_index: npt.NDArray[Any], node_config_ids: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """These features are highly skewed"""
    # log transform feature 28, 120 (shape product, slice_dims_limit_product). this feature is highly skewed.
    x[:, [21, 22, 24, 27, 28, 44, 120, 124]] = np.log(
        x[:, [21, 22, 24, 27, 28, 44, 120, 124]] + 1e-4
    )
    return x, edge_index, node_config_ids


class NodePreprocessor:
    NODE_FEAT_INDEX = np.arange(140)

    def __init__(self, dataset: Literal["xla", "nlp"]):
        if dataset == "xla":
            means, stdevs = (
                XLA_TRAIN_NODE_MEANS[self.NODE_FEAT_INDEX],
                XLA_TRAIN_NODE_STDEVS[self.NODE_FEAT_INDEX],
            )
        else:
            means, stdevs = (
                NLP_TRAIN_NODE_MEANS[self.NODE_FEAT_INDEX],
                NLP_TRAIN_NODE_STDEVS[self.NODE_FEAT_INDEX],
            )

        self.drop_mask = stdevs == 0
        self.norm_mask = np.logical_and(
            ~self.drop_mask, NUMERIC_FEATURE_MASK[self.NODE_FEAT_INDEX]
        )

        self.mean = means[self.norm_mask]
        self.std = stdevs[self.norm_mask]

    def __call__(
        self,
        x: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        x, edge_index, node_config_ids = log_transform_specific_features(
            x, edge_index, node_config_ids
        )

        # normalize node features. intersection of ~DROP_MASK and NUMERIC_FEATURE_MASK
        x[:, self.norm_mask] = (x[:, self.norm_mask] - self.mean) / self.std

        # drop features if they have zero stdev on the train set and are numeric
        x = x[:, ~self.drop_mask]

        return x, edge_index, node_config_ids


class ConfigFeatureGenerator:
    OUTPUT_MASK = np.arange(6)
    INPUT_MASK = np.arange(6, 12)
    KERNEL_MASK = np.arange(12, 18)

    def __call__(
        self,
        config_features: npt.NDArray[Any],  # config is nc x c x 18
    ) -> npt.NDArray[Any]:
        output_features = config_features[:, :, self.OUTPUT_MASK]
        input_features = config_features[:, :, self.INPUT_MASK]
        kernel_features = config_features[:, :, self.KERNEL_MASK]

        # Compute default flags
        output_is_default = np.all(output_features == -1, axis=-1)
        input_is_default = np.all(input_features == -1, axis=-1)
        kernel_is_default = np.all(kernel_features == -1, axis=-1)

        # Get Active Dims (count of non-default dims)
        output_active_dims = np.sum(output_features != -1, axis=-1)
        input_active_dims = np.sum(input_features != -1, axis=-1)
        kernel_active_dims = np.sum(kernel_features != -1, axis=-1)

        # Get max order
        output_max_order = np.max(output_features, axis=-1)
        input_max_order = np.max(input_features, axis=-1)
        kernel_max_order = np.max(kernel_features, axis=-1)

        # Get min order
        output_min_order = np.min(output_features, axis=-1)
        input_min_order = np.min(input_features, axis=-1)
        kernel_min_order = np.min(kernel_features, axis=-1)

        # Get contiguity (# of contiguous dims)
        # Temporarily ignore the -1s
        output_contiguity_count = self.calculate_contiguity(output_features)
        input_contiguity_count = self.calculate_contiguity(input_features)
        kernel_contiguity_count = self.calculate_contiguity(kernel_features)

        # Get contiguity rank (contiguity / active_dims)
        output_contiguity_rank = output_contiguity_count / (output_active_dims + 1e-4)
        input_contiguity_rank = input_contiguity_count / (input_active_dims + 1e-4)
        kernel_contiguity_rank = kernel_contiguity_count / (kernel_active_dims + 1e-4)

        # normalize counts
        output_active_dims = output_active_dims / 6
        input_active_dims = input_active_dims / 6
        kernel_active_dims = kernel_active_dims / 6

        # normalize orders
        output_max_order = output_max_order / 6
        input_max_order = input_max_order / 6
        kernel_max_order = kernel_max_order / 6

        output_min_order = output_min_order / 6
        input_min_order = input_min_order / 6
        kernel_min_order = kernel_min_order / 6

        # normalize contiguity
        output_contiguity_count = output_contiguity_count / 5
        input_contiguity_count = input_contiguity_count / 5
        kernel_contiguity_count = kernel_contiguity_count / 5

        output_input_match = np.all(output_features == input_features, axis=-1)
        output_kernel_match = np.all(output_features == kernel_features, axis=-1)
        input_kernel_match = np.all(input_features == kernel_features, axis=-1)

        # combine all features
        new_config_features = np.stack(
            [
                output_is_default,
                input_is_default,
                kernel_is_default,
                output_active_dims,
                input_active_dims,
                kernel_active_dims,
                output_max_order,
                input_max_order,
                kernel_max_order,
                output_min_order,
                input_min_order,
                kernel_min_order,
                output_contiguity_count,
                input_contiguity_count,
                kernel_contiguity_count,
                output_contiguity_rank,
                input_contiguity_rank,
                kernel_contiguity_rank,
                output_input_match,
                output_kernel_match,
                input_kernel_match,
            ],
            axis=-1,
        )

        # stack with original features
        config_features = np.concatenate(
            [config_features, new_config_features], axis=-1
        )
        return config_features

    def calculate_contiguity(self, features: npt.NDArray[Any]) -> npt.NDArray[Any]:
        valid_mask = features != -1

        # Apply the mask to keep only valid values and fill the rest with `np.nan` for later use with `np.diff`
        valid_features = np.where(valid_mask, features, np.nan)

        # Calculate the difference along the last axis, ignoring NaNs
        diffs = np.diff(valid_features, axis=-1, prepend=np.nan)

        # Check if the differences are 0 or 1 which indicates contiguity
        contiguity_mask = np.isin(diffs, [0, 1])

        # Sum over the last axis to get the count of contiguous elements, ignoring NaNs
        contiguity_count = np.nansum(contiguity_mask, axis=-1)

        return contiguity_count


class GlobalFeatureGenerator:
    def __init__(
        self, dataset: Literal["xla", "nlp"], subtype: Literal["default", "random"]
    ):
        self.dataset = dataset
        self.subtype = subtype

    def __call__(
        self,
        x: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        # Global Features
        # 1. Number of nodes
        # 2. Number of edges
        # 3. Number of configs
        digraph = nx.DiGraph()
        digraph.add_edges_from(edge_index)  # type: ignore

        # These ones I think are justified since we do have this information at inference time
        # 4. Is nlp/xla
        # 5. Is default/random
        # is_nlp = self.dataset == "nlp" # maybe not this one, since we're training on both individually currently
        is_default = self.subtype == "default"
        longest_path_count = nx.dag_longest_path_length(digraph)  # type: ignore
        longest_path_count = cast(float, longest_path_count)
        longest_path_normalized = longest_path_count / x.shape[0]
        component_count = 0

        component_shortest_paths: list[int] = []
        undirected_version = digraph.to_undirected()  # type: ignore
        for C in (
            undirected_version.subgraph(c).copy()
            for c in nx.connected_components(undirected_version)
        ):
            component_shortest_paths.append(nx.average_shortest_path_length(C))
            component_count += 1

        community_rate = component_count / x.shape[0]
        component_shortest_paths_mean = np.mean(component_shortest_paths)
        component_shortest_paths_std = np.std(component_shortest_paths)

        log_num_nodes = np.log(x.shape[0])

        return np.array(
            [
                log_num_nodes,
                is_default,
                longest_path_normalized,
                community_rate,
                component_shortest_paths_mean,
                component_shortest_paths_std,
            ]
        )
