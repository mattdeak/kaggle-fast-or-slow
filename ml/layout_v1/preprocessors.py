from typing import Any

import numpy as np
import numpy.typing as npt
import torch


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
