import numpy as np
import pytest

from ml.layout_v1.preprocessors import (CommunityNodeRemapper, ConfigMetaGraph,
                                        remap_edges, remap_nodes)


def test_community_node_remapper_1hop():
    edge_index = np.array(
        [[0, 1], [1, 2], [1, 3], [2, 4], [2, 5], [3, 6], [5, 7], [6, 8]]
    )

    node_config_ids = np.array([1, 4, 8])
    remapper = CommunityNodeRemapper(1)
    result = remapper(edge_index, node_config_ids)
    expected_result = np.array([0, 1, 2, 3, 4, 6, 8])  # Based on your expected output
    np.testing.assert_array_equal(result, expected_result)


def test_config_meta_graph():
    edge_index = np.array(
        [[0, 1], [1, 2], [1, 3], [2, 4], [2, 5], [3, 6], [5, 7], [6, 8]]
    )
    node_config_ids = np.array([1, 4, 8])
    config_graph = ConfigMetaGraph()
    new_edge_index, weights = config_graph(edge_index, node_config_ids)
    expected_edge_index = np.array([[1, 4], [1, 8]])  # Expected output
    expected_weights = np.array([1 / 2, 1 / 3])  # Assuming each hop has weight 1
    np.testing.assert_array_equal(new_edge_index, expected_edge_index)
    np.testing.assert_array_equal(weights, expected_weights)


def test_remap_nodes():
    node_features = np.array(
        [[0, 1], [1, 2], [1, 3], [2, 4], [2, 5], [3, 6], [5, 7], [6, 8]]
    )
    opcodes = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    node_mapping = np.array([0, 2, 7])
    new_features, new_opcodes = remap_nodes(node_features, opcodes, node_mapping)
    expected_features = np.array([[0, 1], [1, 3], [6, 8]])  # Expected output
    expected_opcodes = np.array([10, 30, 80])  # Expected output
    np.testing.assert_array_equal(new_features, expected_features)
    np.testing.assert_array_equal(new_opcodes, expected_opcodes)


def test_remap_edges_noedges():
    edge_index = np.array(
        [[0, 1], [1, 2], [1, 3], [2, 4], [2, 5], [3, 6], [5, 7], [6, 8]]
    )
    node_mapping = np.array([0, 2, 7])
    new_edge_index, _ = remap_edges(edge_index, node_mapping)

    expected_index = np.array([]).reshape(0, 2)
    np.testing.assert_array_equal(new_edge_index, expected_index)


def test_remap_edges_someedges():
    edge_index = np.array(
        [[0, 1], [1, 2], [1, 3], [2, 4], [2, 5], [3, 6], [5, 7], [6, 8]]
    )
    edge_index_attrs = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    node_mapping = np.array([0, 5, 6, 7, 8])
    new_edge_index, new_index_attrs = remap_edges(
        edge_index, node_mapping, edge_index_attrs
    )

    expected_index = np.array([[1, 3], [2, 4]])  # Expected output
    expected_index_attrs = np.array([7, 8])  # Expected output

    assert new_index_attrs is not None

    np.testing.assert_array_equal(new_edge_index, expected_index)
    np.testing.assert_array_equal(new_index_attrs, expected_index_attrs)
