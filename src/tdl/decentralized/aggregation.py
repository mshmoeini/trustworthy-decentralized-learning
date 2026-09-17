"""Self-plus-neighbor averaging from one immutable post-training snapshot."""

from collections.abc import Mapping, Sequence

import networkx as nx
import torch

from tdl.decentralized.topology import validate_graph
from tdl.federated.aggregation import weighted_average_state_dicts


def aggregate_neighborhoods(
    snapshot: Mapping[int, Mapping[str, torch.Tensor]],
    sample_counts: Mapping[int, int],
    graph: nx.Graph,
    node_order: Sequence[int] | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Return every node's new state without changing any snapshot tensor.

    For node i: sum_{j in {i} union neighbors(i)} n_j / sum_neighborhood(n) * w_j.
    Contributors are sorted by ID for stable floating-point summation order.
    node_order changes only the order results are computed, never their inputs.
    """
    validate_graph(graph, len(snapshot))
    if set(snapshot) != set(graph) or set(sample_counts) != set(graph):
        raise ValueError("Snapshot and sample counts must match graph node IDs.")
    order = sorted(graph) if node_order is None else list(node_order)
    if len(order) != len(graph) or set(order) != set(graph):
        raise ValueError("node_order must contain every node exactly once.")
    updates = {}
    for node in order:
        neighborhood = sorted({node, *graph.neighbors(node)})
        updates[node] = weighted_average_state_dicts(
            [snapshot[neighbor] for neighbor in neighborhood],
            [sample_counts[neighbor] for neighbor in neighborhood],
        )
    return updates
