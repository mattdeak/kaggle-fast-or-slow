from typing import Any, Literal, cast

import networkx as nx
import numpy as np
import numpy.typing as npt


class GlobalFeatureGenerator:
    def __init__(
        self,
        dataset_type: Literal["nlp", "xla"],
        dataset_subtype: Literal["default", "random"],
        subtype_indicator: bool = False,
    ) -> None:
        self.dataset_type = dataset_type
        self.dataset_subtype = dataset_subtype
        self.subtype_indicator = subtype_indicator

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
        is_default = None
        if self.subtype_indicator:
            is_default = self.dataset_subtype == "default"

        longest_path_count = nx.dag_longest_path_length(digraph)  # type: ignore
        longest_path_count = cast(float, longest_path_count)
        longest_path_normalized = longest_path_count / (x.shape[0] + 1)
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

        log_num_nodes = np.log(x.shape[0] + 1e-4)

        final = np.array(
            [
                log_num_nodes,
                longest_path_normalized,
                community_rate,
                component_shortest_paths_mean,
                component_shortest_paths_std,
            ]
        )

        if is_default is not None:
            final = np.append(final, is_default)

        return final

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset_type}, subtype={self.dataset_subtype})"
