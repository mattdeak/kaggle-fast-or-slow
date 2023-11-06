from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ml.layout_v1.stats import (NLP_TRAIN_NODE_MEANS, NLP_TRAIN_NODE_STDEVS,
                                NUMERIC_FEATURE_MASK, XLA_TRAIN_NODE_MEANS,
                                XLA_TRAIN_NODE_STDEVS)


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
        x, edge_index, node_config_ids = _log_transform_specific_features(
            x, edge_index, node_config_ids
        )

        # normalize node features. intersection of ~DROP_MASK and NUMERIC_FEATURE_MASK
        x[:, self.norm_mask] = (x[:, self.norm_mask] - self.mean) / self.std

        # drop features if they have zero stdev on the train set and are numeric
        x = x[:, ~self.drop_mask]

        return x, edge_index, node_config_ids

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"


def _log_transform_specific_features(
    x: npt.NDArray[Any], edge_index: npt.NDArray[Any], node_config_ids: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """These features are highly skewed"""
    # log transform feature 28, 120 (shape product, slice_dims_limit_product). this feature is highly skewed.
    x[:, [21, 22, 24, 27, 28, 44, 120, 124]] = np.log(
        x[:, [21, 22, 24, 27, 28, 44, 120, 124]] + 1e-4
    )
    return x, edge_index, node_config_ids
