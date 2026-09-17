"""Seeded simple, connected, undirected communication graphs."""

from numbers import Integral, Real
import random

import networkx as nx
import numpy as np


TOPOLOGIES = ("ring_degree_2", "ring_degree_4", "random_regular_degree_4",
              "small_world_degree_4", "fully_connected")


def validate_graph(graph: nx.Graph, num_nodes: int, degree: int | None = None) -> None:
    """Reject directed/multi/disconnected graphs, loops, wrong IDs or degrees."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral) or num_nodes < 2:
        raise ValueError("num_nodes must be an integer >= 2.")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Communication graph must be simple and undirected.")
    if set(graph.nodes) != set(range(num_nodes)):
        raise ValueError("Node IDs must be exactly 0..num_nodes-1.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Communication graph must have no self-loops.")
    if not nx.is_connected(graph):
        raise ValueError("Communication graph must be connected.")
    if degree is not None and any(d != degree for _, d in graph.degree()):
        raise ValueError(f"Every node must have degree {degree}.")


def build_topology(name: str, num_nodes: int, seed: int,
                   small_world_rewire_probability: float = 0.2) -> nx.Graph:
    """Construct a topology; random regular connectivity uses bounded retries.

    Rings use modular offsets +/-1 (and +/-2 for degree 4). Watts-Strogatz
    rewiring preserves edge count, not individual degrees. A local Python RNG
    drives random graph generation independently of training/partition RNGs.
    """
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral) or num_nodes < 2:
        raise ValueError("num_nodes must be an integer >= 2.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    p = small_world_rewire_probability
    if isinstance(p, bool) or not isinstance(p, Real) or not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("small_world_rewire_probability must be finite and between 0 and 1.")
    if name not in TOPOLOGIES:
        raise ValueError(f"Unknown topology: {name}.")
    minimum = 2 if name == "fully_connected" else 3 if name == "ring_degree_2" else 5
    if num_nodes < minimum:
        raise ValueError(f"{name} requires at least {minimum} nodes.")
    rng = random.Random(int(seed))
    degree = None
    if name.startswith("ring_"):
        degree = 2 if name == "ring_degree_2" else 4
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from((node, (node + offset) % num_nodes)
                             for node in range(num_nodes) for offset in range(1, degree // 2 + 1))
    elif name == "random_regular_degree_4":
        degree = 4
        for _ in range(100):
            graph = nx.random_regular_graph(4, num_nodes, seed=rng)
            if nx.is_connected(graph):
                break
        else:
            raise RuntimeError("Could not construct a connected regular graph after 100 attempts.")
    elif name == "small_world_degree_4":
        try:
            graph = nx.connected_watts_strogatz_graph(num_nodes, 4, p, tries=100, seed=rng)
        except nx.NetworkXError as error:
            raise RuntimeError("Could not construct a connected small-world graph after 100 attempts.") from error
    else:
        degree = num_nodes - 1
        graph = nx.complete_graph(num_nodes)
    validate_graph(graph, num_nodes, degree)
    return graph


def topology_summary(graph: nx.Graph) -> dict:
    """Describe adjacency and idealized full-model traffic (no bytes/timing)."""
    validate_graph(graph, len(graph))
    degrees = [d for _, d in graph.degree()]
    return {"num_nodes": len(graph), "edges": graph.number_of_edges(),
            "average_degree": sum(degrees) / len(graph), "minimum_degree": min(degrees),
            "maximum_degree": max(degrees), "connected": nx.is_connected(graph),
            "directed_model_transmissions_per_round": 2 * graph.number_of_edges(),
            "average_aggregation_size_including_self": 1 + sum(degrees) / len(graph),
            "edge_list": [list(edge) for edge in sorted(tuple(sorted(edge)) for edge in graph.edges())],
            "neighbors": {str(node): sorted(graph.neighbors(node)) for node in sorted(graph)}}
