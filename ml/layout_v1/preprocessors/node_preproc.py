from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler  # type: ignore

from lib.feature_name_mapping import NODE_FEATURE_IX_LOOKUP

EPSILON = 1e-4


class NodeProcessor:
    NODE_FEAT_INDEX = np.arange(140)
    SHAPE_DIM_FEATURES = np.arange(21, 27)

    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self._standardizer = NodeStandardizer()

    def fit(self, x: npt.NDArray[Any]) -> None:
        if self.standardize:
            self._standardizer.fit(x)

    def __call__(
        self,
        x: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        # normalize node features. intersection of ~DROP_MASK and NUMERIC_FEATURE_MASK
        # Engineered features
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

        if self.standardize:
            x, edge_index, node_config_ids = self._standardizer(
                x, edge_index, node_config_ids
            )

        # log transform specific features
        x = np.hstack((x, engineered))
        return x, edge_index, node_config_ids

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
        ohe_present_threshold: float = 0.1,
    ):
        self.ohe_present_threshold = ohe_present_threshold
        self.standardizer = StandardScaler()
        self._fitted = False

    def fit(self, x: npt.NDArray[Any]) -> None:
        """X is a nc x n x f array"""
        # fit standardizer
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])

        # Detect features with zero variance
        drop_mask = np.var(x, axis=0) == 0

        # Also features where they are clearly one-hot, but the ratio of the
        # positive class is less than the threshold
        features_are_only_zeros_or_ones = np.all(np.isin(x, [0, 1]), axis=0)
        drop_mask = np.logical_or(
            x[:, features_are_only_zeros_or_ones].mean(axis=0)
            > self.ohe_present_threshold,
            drop_mask,
        )
        self._drop_mask = drop_mask

        # We need to re-map the log-transformed indices to the ones that are actually
        # present after we drop
        x = _log_transform_specific_features(x)

        # log transform specific features
        self.standardizer.fit(x[:, ~drop_mask])
        self._fitted = True

    def __call__(
        self,
        x: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        if not self._fitted:
            raise RuntimeError("Standardizer not fitted")

        x = _log_transform_specific_features(x)
        x = x[:, ~self._drop_mask]
        x = self.standardizer.transform(x)

        return (
            x,
            edge_index,
            node_config_ids,
        )


def _log_transform_specific_features(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """These features are highly skewed"""
    # log transform feature 28, 120 (shape product, slice_dims_limit_product). this feature is highly skewed.
    x[:, [21, 22, 24, 27, 28, 44, 115, 120, 124, 127]] = np.log(
        x[:, [21, 22, 24, 27, 28, 44, 115, 120, 124, 127]] + 1e-4
    )
    return x
