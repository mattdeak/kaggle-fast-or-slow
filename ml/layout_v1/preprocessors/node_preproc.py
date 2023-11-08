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
                np.log(self.calculate_shape_sparsity(x) + EPSILON),
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

        # Step 1: Detect features with zero variance and one-hot features
        drop_mask = np.var(x, axis=0) == 0
        ohe_mask = np.all(np.isin(x, [0, 1]), axis=0)

        # Step 2: Detect ohes that are under the threshold
        ohe_under_threshold = ohe_mask & (
            np.mean(x, axis=0) < self.ohe_present_threshold
        )

        # Step 3: Update drop mask with ohe under threshold
        drop_mask = ohe_under_threshold | drop_mask

        self._drop_mask = drop_mask
        self._ohe_mask = ohe_mask  # tells us which features are one-hot

        # Step 4: Log transform specific features as determined by analysis
        x = _log_transform_specific_features(x)

        # Step 5: Fit standardizer on non-drop and non-ohe features
        self.standardizer.fit(x[:, (~drop_mask) | (~ohe_mask)])
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
        standardized = self.standardizer.transform(  #
            x[:, (~self._drop_mask) & (~self._ohe_mask)]
        )
        x[:, (~self._drop_mask) & (~self._ohe_mask)] = standardized
        x = x[:, ~self._drop_mask]

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
