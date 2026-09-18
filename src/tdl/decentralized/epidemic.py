"""EL-Local: independent directed k-out sampling and synchronous snapshot mixing.

Sources: arXiv:2310.01972 and Algorithm 1 of arXiv:2602.03383.
Morph revision 57d73b921c317b0f0d7d9f4a71a6db8051beaf82:
EL_Local.get_neighbors/run and PlainAverageSharing._averaging.
The candidate pool is explicitly all other nodes, as resolved for this project,
not the restricted input-overlay pool in Morph's current tutorial launcher.
"""

import argparse
import json
from numbers import Integral
from pathlib import Path
import random
import statistics

import networkx as nx
import torch

from tdl.config import load_config
from tdl.byzantine.attacks import start_snapshots, outgoing_payloads
from tdl.byzantine.diagnostics import delivery_summary
from tdl.decentralized.runner import evaluate_nodes, prepare_experiment
from tdl.decentralized.training import train_node_snapshots, summarize_local_training
from tdl.federated.aggregation import weighted_average_state_dicts

MODES = ("paper_epidemic", "epidemic_sample_weighted_control")


def validate_protocol(num_nodes, k, seed, round_number):
    for key, value, minimum in [("num_nodes", num_nodes, 2), ("k", k, 1),
                                ("seed", seed, 0), ("round_number", round_number, 1)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    if k >= num_nodes:
        raise ValueError("k must be less than num_nodes.")


def sample_communication(num_nodes, k, seed, round_number):
    """i -> j means i sends to j. Local per-sender RNG; no global RNG use.

    RNG seed = (seed + round_number) * num_nodes + sender_id.
    Exactly k distinct uniform destinations from all other IDs; no repair,
    connectivity conditioning, or symmetrization is applied.
    """
    validate_protocol(num_nodes, k, seed, round_number)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    for sender in range(num_nodes):
        rng = random.Random((seed + round_number) * num_nodes + sender)
        peers = rng.sample([node for node in range(num_nodes) if node != sender], k)
        graph.add_edges_from((sender, peer) for peer in peers)
    return graph


def validate_communication(graph, num_nodes):
    if not graph.is_directed() or graph.is_multigraph() or set(graph) != set(range(num_nodes)) or nx.number_of_selfloops(graph):
        raise ValueError("Use a simple directed graph with IDs 0..n-1 and no self edges.")


def communication_summary(graph, previous=None):
    validate_communication(graph, len(graph))
    incoming = [graph.in_degree(node) for node in sorted(graph)]
    outgoing = [graph.out_degree(node) for node in sorted(graph)]
    zero = [node for node in sorted(graph) if graph.in_degree(node) == 0]
    edges = set(graph.edges())
    old = set(previous.edges()) if previous is not None else None
    return {"directed_edges": len(edges), "transmissions_per_round": len(edges),
            "mean_out_degree": statistics.mean(outgoing), "min_out_degree": min(outgoing),
            "max_out_degree": max(outgoing), "mean_in_degree": statistics.mean(incoming),
            "min_in_degree": min(incoming), "max_in_degree": max(incoming),
            "in_degrees": {str(node): graph.in_degree(node) for node in sorted(graph)},
            "out_degrees": {str(node): graph.out_degree(node) for node in sorted(graph)},
            "zero_in_degree_nodes": zero, "zero_in_degree_count": len(zero),
            "zero_in_degree_fraction": len(zero) / len(graph),
            "weakly_connected": nx.is_weakly_connected(graph),
            "strongly_connected": nx.is_strongly_connected(graph),
            "edge_list": [list(edge) for edge in sorted(edges)],
            "edge_jaccard_previous_round": len(edges & old) / len(edges | old) if old is not None else None,
            "average_aggregation_size_including_self": 1 + statistics.mean(incoming)}


def aggregate_incoming(snapshot, counts, graph, aggregation_mode="paper_epidemic", node_order=None, self_snapshot=None):
    """Read only frozen states; self plus incoming senders, never successors.

    Morph EL_Local.run skips averaging when no model is received. Cloning the
    locally trained snapshot reproduces that behavior without aliasing storage.
    The project's safe buffer policy is reused: integer buffers must agree.
    The Fashion-MNIST CNN has no buffers, so this does not alter these runs.
    """
    validate_communication(graph, len(snapshot))
    if set(snapshot) != set(graph) or set(counts) != set(graph):
        raise ValueError("States and sample counts must match graph IDs.")
    if aggregation_mode not in MODES:
        raise ValueError("Unknown aggregation_mode.")
    order = sorted(graph) if node_order is None else list(node_order)
    if len(order) != len(graph) or set(order) != set(graph):
        raise ValueError("node_order must contain every node exactly once.")
    updates = {}
    own = snapshot if self_snapshot is None else self_snapshot
    if set(own) != set(snapshot):
        raise ValueError("Self states must match sender IDs")
    for node in order:
        senders = sorted({node, *graph.predecessors(node)})
        if len(senders) == 1:
            updates[node] = {key: value.detach().clone() for key, value in own[node].items()}
        else:
            weights = [1 if aggregation_mode == "paper_epidemic" else counts[sender] for sender in senders]
            updates[node] = weighted_average_state_dicts([own[sender] if sender == node else snapshot[sender] for sender in senders], weights)
    return updates


def run_epidemic_round(models, loaders, config, device, round_number, node_order=None):
    validate_protocol(len(models), config["epidemic_k"], config["seed"], round_number)
    if config["aggregation_mode"] not in MODES:
        raise ValueError("Unknown aggregation_mode.")
    if set(models) != set(range(len(models))):
        raise ValueError("Node IDs must be 0..n-1.")
    starts = start_snapshots(models, config)
    snapshot, counts, local_metrics = train_node_snapshots(models, loaders, config, device, round_number, node_order)
    outgoing = outgoing_payloads(starts, snapshot, config) if starts is not None else snapshot
    # Communication sampling follows the post-training snapshot barrier.
    graph = sample_communication(len(models), config["epidemic_k"], config["seed"], round_number)
    updates = aggregate_incoming(outgoing, counts, graph, config["aggregation_mode"], node_order,
                                 **({"self_snapshot": snapshot} if starts is not None else {}))
    # Every next-round state is computed before any live state is installed.
    for node in sorted(models):
        models[node].load_state_dict(updates[node])
    metrics = summarize_local_training(round_number, counts, local_metrics)
    if "byzantine_nodes" in config or "attack" in config:
        metrics["byzantine_deliveries"] = delivery_summary(graph, config.get("byzantine_nodes", []), starts is not None)
    return metrics, graph


def run(config_path, output, alpha=None, k=None, aggregation_mode=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserve the existing report: {output}; choose a new output.")
    config = load_config(config_path)
    for key, value in [("alpha", alpha), ("epidemic_k", k), ("aggregation_mode", aggregation_mode)]:
        if value is not None:
            config[key] = value
    config.setdefault("aggregation_mode", "paper_epidemic")
    config.setdefault("min_samples_per_client", 1)
    config.setdefault("num_workers", 0)
    validate_protocol(config["num_clients"], config["epidemic_k"], config["seed"], 1)
    if config["aggregation_mode"] not in MODES:
        raise ValueError("Unknown aggregation_mode.")
    for key, minimum in [("rounds", 1), ("local_epochs", 1), ("batch_size", 1),
                         ("num_workers", 0), ("min_samples_per_client", 1)]:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    models, loaders, test_loader, device, partition, train_size, test_size = prepare_experiment(config)
    initial = evaluate_nodes(models, test_loader, device)
    for node in initial["nodes"]:
        node.update(local_training_loss=None, num_local_training_samples=len(loaders[node["node_id"]].dataset))
    rule = "uniform_self_plus_incoming" if config["aggregation_mode"] == "paper_epidemic" else "sample_count_weighted_self_plus_incoming"
    report = {"metadata": {"algorithm": config["aggregation_mode"], "topology_type": "dynamic_directed_random_k_out",
              "aggregation_rule": rule, "dataset": "Fashion-MNIST", "device": str(device),
              "training_samples": train_size, "test_samples": test_size,
              "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda, "networkx_version": nx.__version__,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "candidate_pool": "all other nodes", "peer_sampling": "k distinct peers uniformly without replacement per sender per round",
              "topology_seed_rule": "(experiment_seed + one_based_round_number) * num_nodes + sender_id; local Python Random",
              "node_seed_rule": "seed + round_number * num_clients + node_id (one-based rounds)",
              "zero_in_degree_behavior": "retain locally trained state; skip mixing",
              "aggregation_formula": "sum_{j in self+incoming} theta_j / (1+in_degree)" if rule.startswith("uniform") else "sum_{j in self+incoming} n_j*theta_j / sum_{j in self+incoming} n_j",
              "comparison_caveat": "Paper EL-Local vs static baselines changes both topology dynamics and aggregation weighting; weighted EL is an explicitly separate control.",
              "source_papers": ["https://arxiv.org/abs/2310.01972", "https://arxiv.org/abs/2602.03383"],
              "source_repository": "https://github.com/bacox/Morph", "source_revision": "57d73b921c317b0f0d7d9f4a71a6db8051beaf82",
              "source_functions": ["EL_Local.get_neighbors", "EL_Local.run", "PlainAverageSharing._averaging"],
              "intentional_differences": ["all-other-node candidate pool as explicitly resolved; current tutorial overlay restriction not reproduced", "Fashion-MNIST CNN, 10 sequential nodes, CUDA, full local epochs, project optimizer reset and seeds", "safe compatible nonfloating buffers; current CNN has none"],
              "consensus_formula": "mean_{i<j} sqrt(sum_p (theta_i,p-theta_j,p)^2 / P); floating named parameters only, buffers excluded",
              "accuracy_std": "population standard deviation across nodes, fraction units; not seed uncertainty",
              "optimizer_state": "fresh per node per round", "simulation": "sequential synchronous",
              "traffic_definition": "n*k directed full-model transmissions per round, excludes self and coordination messages",
              "development_only": True, "interpretation": "Single seed, three-round development checks; no confidence intervals or significance claims; not paper experiment replication."},
              "config": config, "partition": partition, "initial_node_evaluation": initial, "rounds": []}
    previous = None
    print(f"Device: {device} | EL-Local k={config['epidemic_k']} | alpha={config['alpha']} | mode={config['aggregation_mode']}", flush=True)
    for number in range(1, config["rounds"] + 1):
        local, graph = run_epidemic_round(models, loaders, config, device, number)
        network = evaluate_nodes(models, test_loader, device)
        evaluations = {node["node_id"]: node for node in network.pop("nodes")}
        topology = communication_summary(graph, previous)
        zero = topology["zero_in_degree_nodes"]
        for node in local["nodes"]:
            node.update(evaluations[node["node_id"]])
            node.update(in_degree=graph.in_degree(node["node_id"]), out_degree=graph.out_degree(node["node_id"]))
        zero_metrics = [{"node_id": node, "test_accuracy": evaluations[node]["test_accuracy"]} for node in zero]
        others = [evaluations[node]["test_accuracy"] for node in sorted(graph) if node not in zero]
        metrics = {**local, **network, "topology": topology, "aggregation_rule": rule,
                   "directed_model_transmissions": graph.number_of_edges(),
                   "zero_in_degree_node_accuracies": zero_metrics,
                   "zero_in_degree_mean_test_accuracy": statistics.mean(n["test_accuracy"] for n in zero_metrics) if zero_metrics else None,
                   "nonzero_in_degree_mean_test_accuracy": statistics.mean(others) if others else None}
        report["rounds"].append(metrics)
        previous = graph
        print(f"Round {number} | mean={metrics['mean_node_test_accuracy']:.2%} | worst={metrics['worst_node_test_accuracy']:.2%} | std={metrics['node_test_accuracy_std']:.2%} | disagreement={metrics['mean_pairwise_rms_parameter_distance']:.8f} | zero incoming={len(zero)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/epidemic.yaml"))
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--k", type=int)
    parser.add_argument("--aggregation-mode", choices=MODES)
    parser.add_argument("--output", type=Path, default=Path("results/epidemic_summary.json"))
    args = parser.parse_args()
    run(args.config, args.output, args.alpha, args.k, args.aggregation_mode)


if __name__ == "__main__":
    main()
