from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from ml.layout_v1.stats import (NUMERIC_FEATURE_MASK, XLA_TRAIN_NODE_MEANS,
                                XLA_TRAIN_NODE_STDEVS)


# These features have zero stdev on the train sets
def reduce_to_config_node_communities_tensor(
    x: torch.Tensor, edge_index: torch.Tensor, node_config_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, _ = torch.where(torch.isin(edge_index, node_config_ids))

    new_edge_index = edge_index[rows]
    kept_nodes = torch.unique(new_edge_index)

    for i, n in enumerate(kept_nodes):
        new_edge_index[new_edge_index == n] = i

    x = x[kept_nodes, :]
    return x, new_edge_index


def reduce_to_config_node_communities_ndarray(
    x: npt.NDArray[Any], edge_index: npt.NDArray[Any], node_config_ids: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    rows, _ = np.where(np.isin(edge_index, node_config_ids))

    new_edge_index = edge_index[rows]
    new_node_config_ids = node_config_ids.copy()
    kept_nodes = np.unique(new_edge_index)

    for i, n in enumerate(kept_nodes):
        new_edge_index[new_edge_index == n] = i
        new_node_config_ids[new_node_config_ids == n] = i

    x = x[kept_nodes, :]
    return x, new_edge_index, new_node_config_ids


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


class XLANodePreprocessor:
    NODE_FEAT_INDEX = np.arange(140)
    DROP_MASK = XLA_TRAIN_NODE_STDEVS == 0
    NORM_MASK = np.logical_and(~DROP_MASK, NUMERIC_FEATURE_MASK)

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

        x[:, self.NORM_MASK] = (
            x[:, self.NORM_MASK] - XLA_TRAIN_NODE_MEANS[self.NORM_MASK]
        ) / XLA_TRAIN_NODE_STDEVS[self.NORM_MASK]

        # drop features if they have zero stdev on the train set
        x = x[:, ~self.DROP_MASK]

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
        output_contiguity_count = np.sum(
            np.isin(np.diff(output_features[output_features != -1.0], axis=-1), [0, 1]),
            axis=-1,
        )
        input_contiguity_count = np.sum(
            np.isin(np.diff(input_features[input_features != -1.0], axis=-1), [0, 1]),
            axis=-1,
        )
        kernel_contiguity_count = np.sum(
            np.isin(np.diff(kernel_features[kernel_features != -1.0], axis=-1), [0, 1]),
            axis=-1,
        )

        # TODO: this is super heavy. can we do this more efficiently?
        # output_nondefault_unique_count = np.sum(
        #     np.unique(output_features, axis=-1) != -1, axis=-1
        # )
        # input_nondefault_unique_count = np.sum(
        #     np.unique(input_features, axis=-1) != -1, axis=-1
        # )
        # kernel_nondefault_unique_count = np.sum(
        #     np.unique(kernel_features, axis=-1) != -1, axis=-1
        # )

        # Get contiguity rank (contiguity / active_dims)
        output_contiguity_rank = output_contiguity_count / (output_active_dims + 1e-4)
        input_contiguity_rank = input_contiguity_count / (input_active_dims + 1e-4)
        kernel_contiguity_rank = kernel_contiguity_count / (kernel_active_dims + 1e-4)

        # normalize counts
        output_active_dims = output_active_dims / 6
        input_active_dims = input_active_dims / 6
        kernel_active_dims = kernel_active_dims / 6

        # output_nondefault_unique_count = output_nondefault_unique_count / 6
        # input_nondefault_unique_count = input_nondefault_unique_count / 6
        # kernel_nondefault_unique_count = kernel_nondefault_unique_count / 6

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
