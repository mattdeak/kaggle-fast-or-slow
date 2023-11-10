from dataclasses import dataclass
from typing import Any, Protocol, cast

import networkx as nx
import numpy as np
import numpy.typing as npt


@dataclass
class GraphTransformReturnType:
    node_features: npt.NDArray[Any]
    opcodes: npt.NDArray[Any]
    edge_index: npt.NDArray[Any]
    node_config_ids: npt.NDArray[Any]
    alternate_edge_index: npt.NDArray[Any]
    edge_index_attr: npt.NDArray[Any] | None = None
    alternate_edge_index_attr: npt.NDArray[Any] | None = None


# Node Features, Opcodes, Edge Index, Node Config Ids, Alternate Edge Index


class GraphTransform(Protocol):
    def __call__(
        self,
        node_features: npt.NDArray[Any],
        opcodes: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> GraphTransformReturnType:
        """Perform a transform on the data (features or edges or both). Return the transformed data."""
        ...


class GraphProcessor:
    """The graph processor by default provides the following:

    1. Shrink the graph to only the nodes that are in the same neighborhood as at least one configurable node.
    2. Provide an additional edge index that connects configurable nodes weighted by the reciprocal of the number of hops between them.
    """

    def __init__(self, hops: int = 1, normalize_by_hops: bool = True):
        self.hops = hops
        self.remapper = CommunityNodeRemapper(number_of_hops=hops)
        self.alt_graph_transform = ConfigMetaGraph(normalize_by_hops=normalize_by_hops)

    def __call__(
        self,
        node_features: npt.NDArray[Any],
        opcodes: npt.NDArray[Any],
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> GraphTransformReturnType:
        """Reduce the graph to only the nodes that are in the same neighborhood as at least one configurable node.
        Args:
            node_features: The node features (n x f)
            opcodes: The opcodes (n)
            edge_index: The edge index in COO (e x 2)
            node_config_ids: The node config ids (nc). Each node config corresponds to a node in the graph where the id is the index of the node.
        Returns:
            The reduced node features, opcodes, edge index, and node config ids.
        """
        alternate_edge_index, alternate_edge_index_attr = self.alt_graph_transform(
            edge_index, node_config_ids
        )

        # Remap the nodes in the edge index.
        node_mapping = self.remapper(edge_index, node_config_ids)
        new_edge_index, _ = remap_edges(edge_index, node_mapping)
        alternate_edge_index, alternate_edge_index_attr = remap_edges(
            alternate_edge_index, node_mapping, alternate_edge_index_attr
        )
        node_features, opcodes = remap_nodes(node_features, opcodes, node_mapping)

        return GraphTransformReturnType(
            node_features=node_features,
            opcodes=opcodes,
            edge_index=new_edge_index,
            node_config_ids=node_config_ids,
            alternate_edge_index=alternate_edge_index,
            edge_index_attr=None,
            alternate_edge_index_attr=alternate_edge_index_attr,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hops={self.hops})"


class CommunityNodeRemapper:
    def __init__(self, number_of_hops: int = 1):
        self.number_of_hops = number_of_hops

    def __call__(
        self,
        edge_index: npt.NDArray[Any],
        node_config_ids: npt.NDArray[Any],
    ) -> npt.NDArray[np.int_]:
        """Based on an edge index and node config IDS, we create a graph
        that contains only the nodes that are in the same neighborhood as
        at least one configurable node. We then remap the node ids to be
        contiguous.
        """
        rows, _ = np.where(np.isin(edge_index, node_config_ids))
        new_edge_index = edge_index[rows]
        kept_nodes = np.unique(new_edge_index)

        # Get nodes further away by iteratively adding neighbors of current nodes to
        # the set of kept nodes.
        for _ in range(self.number_of_hops - 1):
            rows, _ = np.where(np.isin(edge_index, kept_nodes))
            new_edge_index = edge_index[rows]
            kept_nodes = np.unique(new_edge_index)

        return kept_nodes


class ConfigMetaGraph:
    """This class creates an additional edge index that connects configurable nodes weighted by the reciprocal of the number of
    hops between them."""

    def __init__(self, normalize_by_hops: bool = True):
        self.normalize_by_hops = normalize_by_hops

    def __call__(
        self, edge_index: npt.NDArray[Any], node_config_ids: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Create an additional edge index that connects configurable nodes weighted by the reciprocal of the number of
        hops between them.
        Args:
            edge_index: The edge index in COO (e x 2)
            node_config_ids: The node config ids (nc). Each node config corresponds to a node in the graph where the id is the index of the node.
        Returns:
            The additional edge index in COO (e x 2)
        """
        # Create a graph from the edge index.
        g = nx.DiGraph()
        g.add_edges_from(edge_index)

        # Get the shortest path lengths between all pairs of nodes.

        # Construct the additional edge index.
        new_edge_index: list[tuple[int, int]] = []
        new_edge_weights: list[float] = []
        for n in node_config_ids:
            for m in node_config_ids:
                #  get the shortest path length between the two nodes
                try:
                    path_length: int = cast(int, nx.shortest_path_length(g, n, m))
                    if path_length == 0:
                        continue

                    new_edge_index.append((n, m))
                    new_edge_weights.append(
                        1 / path_length
                    ) if self.normalize_by_hops else new_edge_weights.append(1)

                except nx.NetworkXNoPath:
                    continue

        return np.array(new_edge_index), np.array(new_edge_weights)


def remap_nodes(
    node_features: npt.NDArray[np.float_],
    opcodes: npt.NDArray[np.int_],
    node_mapping: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    """Remaps the node features and opcodes to the new node ids.

    Args:
        node_features: The node features (n x f), indexed by node id
        opcodes: The opcodes (n), indexed by node id
        node_mapping: The mapping from old node ids to new node ids (n) (node at ix becomes node at node_mapping[ix])
    """

    # node features and opcodes are indexed by node id
    new_node_features = node_features[node_mapping, :]
    new_opcodes = opcodes[node_mapping]

    return new_node_features, new_opcodes


def remap_edges(
    edge_index: npt.NDArray[np.int_],
    node_mapping: npt.NDArray[np.int_],
    edge_index_attr: npt.NDArray[np.float_] | None = None,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float_] | None]:
    """Remaps the edge index to the new node ids.
    Args:
        edge_index: The edge index in COO (e x 2)
        node_mapping: The mapping from old node ids to new node ids (n). Not all nodes in the edge index are guaranteed to be in the node mapping.
                        In this case, we remove the edge from the edge index.
    Returns:
        The remapped edge index in COO (e x 2)
    """
    # We can only remap the edge index if both nodes are in the node mapping.
    # Otherwise, we remove the edge.
    ix = np.isin(edge_index[:, 0], node_mapping) & np.isin(
        edge_index[:, 1], node_mapping
    )

    new_edge_index = edge_index[ix, :]

    new_edge_index_attr = None
    if edge_index_attr is not None:
        new_edge_index_attr = edge_index_attr[ix]

    for i, n in enumerate(node_mapping):
        new_edge_index[new_edge_index == n] = i

    return new_edge_index, new_edge_index_attr
