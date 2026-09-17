"""Synchronous multi-topology learning on independently owned node models."""

import argparse
import json
from numbers import Integral
from pathlib import Path
import statistics
from typing import Any

import networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from tdl.config import load_config
from tdl.data.fashion_mnist import load_fashion_mnist
from tdl.data.inspect_partition import summarize_partition
from tdl.data.partition import create_dirichlet_partition
from tdl.decentralized.consensus import mean_pairwise_rms_parameter_distance
from tdl.decentralized.topology import TOPOLOGIES, build_topology, topology_summary
from tdl.decentralized.training import initialize_node_models, run_decentralized_round
from tdl.models import FashionMNISTCNN
from tdl.seed import set_seed
from tdl.training.centralized import resolve_device
from tdl.training.evaluate import evaluate


def evaluate_nodes(models: dict[int, nn.Module], loader: DataLoader,
                   device: torch.device) -> dict[str, Any]:
    """Evaluate the same test set per node; summarize nodes with equal weights.

    Accuracy std is population std across this network's nodes, not uncertainty
    across independent runs. Models return to CPU for storage/consensus metrics.
    """
    metrics = []
    for node, model in sorted(models.items()):
        model.to(device)
        loss, accuracy = evaluate(model, loader, nn.CrossEntropyLoss(), device)
        model.cpu()
        metrics.append({"node_id": node, "test_loss": loss, "test_accuracy": accuracy})
    accuracies = [node["test_accuracy"] for node in metrics]
    return {"nodes": metrics, "mean_node_test_accuracy": statistics.mean(accuracies),
            "worst_node_test_accuracy": min(accuracies), "best_node_test_accuracy": max(accuracies),
            "node_test_accuracy_std": statistics.pstdev(accuracies),
            "mean_node_test_loss": statistics.mean(node["test_loss"] for node in metrics),
            "mean_pairwise_rms_parameter_distance": mean_pairwise_rms_parameter_distance(models)}


def prepare_experiment(config):
    """Shared seeded data, partitions, loaders, and independent initial models."""
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    train_data, test_data = load_fashion_mnist(config["data_dir"])
    partition = create_dirichlet_partition(train_data.targets.numpy(), config["num_clients"],
                                           config["alpha"], config["seed"], config["min_samples_per_client"])
    partition_summary = summarize_partition(train_data.targets.numpy(), partition)
    if not partition_summary["coverage_complete"]:
        raise RuntimeError("Training partition must cover every sample exactly once.")
    options = {"batch_size": config["batch_size"], "num_workers": config["num_workers"],
               "pin_memory": device.type == "cuda"}
    loaders = {node: DataLoader(Subset(train_data, indices), shuffle=True,
                                generator=torch.Generator().manual_seed(config["seed"] + node), **options)
               for node, indices in partition.items()}
    test_loader = DataLoader(test_data, shuffle=False,
                             generator=torch.Generator().manual_seed(config["seed"]), **options)
    # The temporary initializer is discarded; no server/global model persists.
    models = initialize_node_models(FashionMNISTCNN(), config["num_clients"])
    return models, loaders, test_loader, device, partition_summary, len(train_data), len(test_data)


def run(config_path: str | Path, output: str | Path, alpha: float | None = None,
        topology: str | None = None) -> dict[str, Any]:
    """Initialize identical nodes, run synchronous rounds, and save local JSON."""
    config = load_config(config_path)
    if alpha is not None:
        config["alpha"] = alpha
    if topology is not None:
        config["topology"] = topology
    config.setdefault("min_samples_per_client", 1)
    config.setdefault("num_workers", 0)
    config.setdefault("small_world_rewire_probability", 0.2)
    for key, minimum in (("rounds", 1), ("local_epochs", 1), ("batch_size", 1),
                         ("num_workers", 0), ("min_samples_per_client", 1)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    graph = build_topology(config["topology"], config["num_clients"], config["seed"],
                           config["small_world_rewire_probability"])
    communication = topology_summary(graph)
    models, loaders, test_loader, device, partition_summary, training_samples, test_samples = prepare_experiment(config)
    initial = evaluate_nodes(models, test_loader, device)
    for node in initial["nodes"]:
        node.update({"local_training_loss": None,
                     "num_local_training_samples": len(loaders[node["node_id"]].dataset)})
    report = {
        "metadata": {"dataset": "Fashion-MNIST", "device": str(device),
                     "training_samples": training_samples, "test_samples": test_samples,
                     "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
                     "networkx_version": nx.__version__,
                     "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "node_seed_rule": "seed + round_number * num_clients + node_id (one-based rounds)",
                     "optimizer_state": "fresh per node per round", "simulation": "sequential synchronous",
                     "aggregation": "self + neighbors, weighted by local partition sample counts",
                     "consensus_formula": "mean_{i<j} sqrt(sum_p ||theta_i,p-theta_j,p||^2 / P); floating parameters only",
                     "accuracy_std": "population standard deviation across nodes, in fraction units",
                     "traffic_definition": "2 * undirected edges full-model transmissions per round; excludes self",
                     "development_only": True},
        "config": config, "topology": communication, "partition": partition_summary,
        "initial_node_evaluation": initial, "rounds": [],
    }
    print(f"Device: {device} | topology: {config['topology']} | alpha: {config['alpha']} | seed: {config['seed']}")
    print(f"Edges: {communication['edges']} | transmissions/round: {communication['directed_model_transmissions_per_round']}")
    print(f"Initial | mean accuracy: {initial['mean_node_test_accuracy']:.2%} | "
          f"disagreement: {initial['mean_pairwise_rms_parameter_distance']:.8f}")
    for round_number in range(1, config["rounds"] + 1):
        local = run_decentralized_round(models, loaders, graph, config, device, round_number)
        network = evaluate_nodes(models, test_loader, device)
        node_evaluations = {item["node_id"]: item for item in network.pop("nodes")}
        for node in local["nodes"]:
            node.update(node_evaluations[node["node_id"]])
        metrics = {**local, **network,
                   "directed_model_transmissions": communication["directed_model_transmissions_per_round"]}
        report["rounds"].append(metrics)
        print(f"Round {round_number}/{config['rounds']} | mean accuracy: {metrics['mean_node_test_accuracy']:.2%} | "
              f"worst: {metrics['worst_node_test_accuracy']:.2%} | std: {metrics['node_test_accuracy_std']:.2%} | "
              f"mean loss: {metrics['mean_node_test_loss']:.4f} | "
              f"disagreement: {metrics['mean_pairwise_rms_parameter_distance']:.8f}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report saved to {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--topology", choices=TOPOLOGIES)
    parser.add_argument("--output", type=Path, default=Path("results/decentralized_summary.json"))
    args = parser.parse_args()
    run(args.config, args.output, args.alpha, args.topology)


if __name__ == "__main__":
    main()
