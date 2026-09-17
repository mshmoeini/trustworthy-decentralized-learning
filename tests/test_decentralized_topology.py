"""Topology invariants and seeded random graph construction."""

import networkx as nx
import pytest

from tdl.decentralized.topology import TOPOLOGIES, build_topology, topology_summary, validate_graph


@pytest.mark.parametrize("name,degree,edges", [("ring_degree_2", 2, 10), ("ring_degree_4", 4, 20),
    ("random_regular_degree_4", 4, 20), ("fully_connected", 9, 45)])
def test_exact_degrees_connectedness_ids_and_traffic(name, degree, edges):
    graph = build_topology(name, 10, 42)
    assert set(graph) == set(range(10))
    assert nx.is_connected(graph) and not graph.is_directed()
    assert nx.number_of_selfloops(graph) == 0
    assert all(d == degree for _, d in graph.degree())
    summary = topology_summary(graph)
    assert summary["edges"] == edges
    assert summary["directed_model_transmissions_per_round"] == 2 * edges
    assert summary["average_aggregation_size_including_self"] == degree + 1


def test_ring_neighbors():
    assert set(build_topology("ring_degree_2", 10, 42).neighbors(0)) == {1, 9}
    assert set(build_topology("ring_degree_4", 10, 42).neighbors(0)) == {1, 2, 8, 9}


@pytest.mark.parametrize("name", ["random_regular_degree_4", "small_world_degree_4"])
def test_random_graph_reproducibility_and_different_seeds(name):
    first = topology_summary(build_topology(name, 10, 42))
    assert first == topology_summary(build_topology(name, 10, 42))
    assert first["edge_list"] != topology_summary(build_topology(name, 10, 43))["edge_list"]


@pytest.mark.parametrize("p", [0.0, 0.2, 1.0])
def test_small_world_degree_is_nominal_and_edges_preserved(p):
    graph = build_topology("small_world_degree_4", 10, 42, p)
    assert nx.is_connected(graph) and nx.number_of_selfloops(graph) == 0
    assert graph.number_of_edges() == 20
    assert topology_summary(graph)["average_degree"] == 4


@pytest.mark.parametrize("name,n", [("ring_degree_2", 2), ("ring_degree_4", 4),
    ("random_regular_degree_4", 4), ("small_world_degree_4", 4), ("fully_connected", 1)])
def test_too_few_nodes(name, n):
    with pytest.raises(ValueError):
        build_topology(name, n, 42)


@pytest.mark.parametrize("kwargs", [{"num_nodes": 1.5}, {"seed": -1},
    {"small_world_rewire_probability": -0.1}, {"small_world_rewire_probability": float("nan")},
    {"small_world_rewire_probability": 1.1}, {"name": "unknown"}])
def test_invalid_topology_options(kwargs):
    with pytest.raises(ValueError):
        build_topology(**{"name": "fully_connected", "num_nodes": 10, "seed": 42, **kwargs})


@pytest.mark.parametrize("graph", [nx.DiGraph([(0, 1)]), nx.MultiGraph([(0, 1)]),
    nx.Graph([(0, 0), (0, 1)]), nx.empty_graph(2), nx.Graph([(1, 2)])])
def test_invalid_graphs(graph):
    with pytest.raises(ValueError):
        validate_graph(graph, 2)
