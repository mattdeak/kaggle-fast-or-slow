from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from lib.feature_name_mapping import NODE_FEATURE_IX_LOOKUP
from ml.layout_v1.stats import (NLP_TRAIN_NODE_MEANS, NLP_TRAIN_NODE_STDEVS,
                                NUMERIC_FEATURE_MASK, XLA_TRAIN_NODE_MEANS,
                                XLA_TRAIN_NODE_STDEVS)

EPSILON = 1e-4


class NodeProcessor:
    NODE_FEAT_INDEX = np.arange(140)
    SHAPE_DIM_FEATURES = np.arange(21, 27)

    def __init__(
        self, dataset: Literal["xla", "nlp"], use_engineered_features: bool = True
    ):
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

        self.use_engineered_features = use_engineered_features
        self.dataset = dataset
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
        # normalize node features. intersection of ~DROP_MASK and NUMERIC_FEATURE_MASK
        # Engineered features
        if self.use_engineered_features:
            engineered = np.hstack(
                (
                    self.calculate_shape_sparsity(x),
                    self.dimensionality(x),
                    self.stride_interactions(x),
                    self.padding_proportions(x),
                    self.effective_window(x),
                    self.reversal_ratio(x),
                )
            )

            # log transform specific features
            x[:, self.norm_mask] = (x[:, self.norm_mask] - self.mean) / self.std
            x = x[:, ~self.drop_mask]
            x = np.hstack((x, engineered))

        # drop features if they have zero stdev on the train set and are numeric
        else:
            x = _log_transform_specific_features(x)
            x[:, self.norm_mask] = (x[:, self.norm_mask] - self.mean) / self.std
            x = x[:, ~self.drop_mask]

        return x, edge_index, node_config_ids

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"

    def calculate_shape_sparsity(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculate shape sparsity of node features. This is the ratio of the"""
        # shape sparsity is the number of non-default features
        # divided by the total number of features
        product = x[:, NODE_FEATURE_IX_LOOKUP["shape_dimensions_product"]]
        sum_dims = x[:, NODE_FEATURE_IX_LOOKUP["shape_dimensions_sum"]]

        return (sum_dims / product).reshape(-1, 1)

    def dimensionality(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculate dimensionality of node features shape"""
        # dimensionality is the number of non-default features
        # divided by the total number of features
        return (np.sum(x[:, self.SHAPE_DIM_FEATURES] != -1, axis=-1) / 6).reshape(-1, 1)

    def stride_interactions(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculate stride interactions of node features shape"""
        return x[:, 37:43] * x[:, 45:51]

    def padding_proportions(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculate padding proportions of node features shape"""
        return x[:, 53:59] / (x[:, 37:43] + EPSILON)

    def effective_window(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculate effective window of node features shape"""
        return (x[:, 37:43] - 1) * x[:, 45:51] + 1

    def reversal_ratio(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculate reversal ratio of node features shape"""
        return (x[:, 91] / (x[:, 91:93].sum(axis=1) + EPSILON)).reshape(-1, 1)


class NodeStandardizer:
    def __init__(
        self,
        run_log_transform: bool = True,
        drop_strategy: Literal["none", "auto"] = "auto",
    ):
        self.log_transform = log_transform
        self.drop_strategy = drop_strategy


def _log_transform_specific_features(
    x: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """These features are highly skewed"""
    # log transform feature 28, 120 (shape product, slice_dims_limit_product). this feature is highly skewed.
    x[:, [21, 22, 24, 27, 28, 44, 115, 120, 124, 127]] = np.log(
        x[:, [21, 22, 24, 27, 28, 44, 115, 120, 124, 127]] + 1e-4
    )
    return x
