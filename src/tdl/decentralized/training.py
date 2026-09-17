"""Sequential local updates with synchronous graph-neighborhood mixing."""

from copy import deepcopy
from numbers import Integral
from typing import Any

import networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader

from tdl.decentralized.aggregation import aggregate_neighborhoods
from tdl.decentralized.topology import validate_graph
from tdl.federated.training import train_client
from tdl.seed import set_seed


def initialize_node_models(initial_model: nn.Module, num_nodes: int) -> dict[int, nn.Module]:
    """Copy identical initial parameters into independently owned node models."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral) or num_nodes < 2:
        raise ValueError("num_nodes must be an integer >= 2.")
    return {node: deepcopy(initial_model) for node in range(num_nodes)}


def train_node_snapshots(models, loaders, config, device, round_number, node_order=None):
    """Train isolated copies and return owned, frozen states plus local metrics."""
    if set(loaders) != set(models):
        raise ValueError("Models and loaders must have the same node IDs.")
    if len({id(model) for model in models.values()}) != len(models):
        raise ValueError("Every node must own a separate model instance.")
    generators = [loader.generator for loader in loaders.values()]
    if any(generator is None for generator in generators) or len({id(g) for g in generators}) != len(models):
        raise ValueError("Every node loader must have an independent generator.")
    if isinstance(round_number, bool) or not isinstance(round_number, Integral) or round_number <= 0:
        raise ValueError("round_number must be a positive integer.")
    order = sorted(models) if node_order is None else list(node_order)
    if len(order) != len(models) or set(order) != set(models):
        raise ValueError("node_order must contain every node exactly once.")
    snapshot, counts, local_metrics = {}, {}, {}
    for node in order:
        seed = config["seed"] + round_number * len(models) + node
        set_seed(seed)
        loaders[node].generator.manual_seed(seed)
        local_model = deepcopy(models[node]).to(device)
        local_metrics[node] = train_client(local_model, loaders[node], config, device)
        counts[node] = local_metrics[node]["num_samples"]
        snapshot[node] = {key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()}
        del local_model
    return snapshot, counts, local_metrics


def summarize_local_training(round_number, counts, local_metrics):
    """Common local-training report; losses are sample weighted for reporting."""
    total = sum(counts.values())
    return {"round": round_number, "num_nodes": len(counts), "total_training_samples": total,
            "weighted_mean_local_training_loss": sum(
                local_metrics[node]["training_loss"] * counts[node] / total for node in sorted(counts)),
            "nodes": [{"node_id": node, "local_training_loss": local_metrics[node]["training_loss"],
                       "local_epoch_losses": local_metrics[node]["epoch_losses"],
                       "num_local_training_samples": counts[node]} for node in sorted(counts)]}


def run_decentralized_round(models: dict[int, nn.Module], loaders: dict[int, DataLoader],
                            graph: nx.Graph, config: dict[str, Any], device: torch.device,
                            round_number: int, node_order: list[int] | None = None) -> dict:
    """Train copies, freeze all states, compute all updates, then install them.

    The live node models remain untouched during training and aggregation.
    Deep copies also avoid partial live-node updates if local training fails.
    CPU snapshots own their storage. No persistent global model is involved.
    Seeds match FedAvg: seed + round_number * num_nodes + node_id.
    Optimizers are reset per node/round, as in the FedAvg baseline.
    """
    validate_graph(graph, len(models))
    if set(models) != set(graph):
        raise ValueError("Models must match graph node IDs.")
    order = sorted(models) if node_order is None else list(node_order)
    snapshot, counts, local_metrics = train_node_snapshots(models, loaders, config, device, round_number, order)
    # Barrier: every trained node state is frozen before any mixing begins.
    updates = aggregate_neighborhoods(snapshot, counts, graph, node_order=order)
    # Barrier: every new state exists before any live model is replaced.
    for node in sorted(models):
        models[node].load_state_dict(updates[node])
    return summarize_local_training(round_number, counts, local_metrics)
