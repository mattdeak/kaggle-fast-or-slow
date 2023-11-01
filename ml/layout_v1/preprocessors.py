import torch


def reduce_to_config_node_communities(
    x: torch.Tensor, edge_index: torch.Tensor, node_config_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, _ = torch.where(torch.isin(edge_index, node_config_ids))

    new_edge_index = edge_index[rows]
    kept_nodes = torch.unique(new_edge_index)

    for i, n in enumerate(kept_nodes):
        new_edge_index[new_edge_index == n] = i

    x = x[kept_nodes, :]
    return x, new_edge_index
