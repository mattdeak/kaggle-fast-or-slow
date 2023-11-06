from typing import Any

import numpy as np
import numpy.typing as npt


class ConfigNodeCommunityPreprocessor:
    """This class provides a callable that reduces the graph to only the nodes that are in the same neighborhood as
    at least one configurable node."""

    def __init__(self, hops: int = 1):
        self.hops = hops

    def __call__(
        self,
        node_features: npt.NDArray[Any],
        opcodes: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        ...
        rows, _ = np.where(np.isin(edge_index, node_config_ids))
        new_edge_index = edge_index[rows]
        kept_nodes = np.unique(new_edge_index)

        for _ in range(self.hops):
            rows, _ = np.where(np.isin(edge_index, kept_nodes))
            new_edge_index = edge_index[rows]
            kept_nodes = np.unique(new_edge_index)

        new_node_config_ids = node_config_ids.copy()

        for i, n in enumerate(kept_nodes):
            new_edge_index[new_edge_index == n] = i
            new_node_config_ids[new_node_config_ids == n] = i

        node_features = node_features[kept_nodes, :]
        opcodes = opcodes[kept_nodes]

        return node_features, opcodes, new_edge_index, new_node_config_ids

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hops={self.hops})"
