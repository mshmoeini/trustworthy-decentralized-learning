"""Reproducible sequential FedAvg on Fashion-MNIST client partitions."""

import argparse
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from tdl.config import load_config
from tdl.data.fashion_mnist import load_fashion_mnist
from tdl.data.inspect_partition import summarize_partition
from tdl.data.partition import create_dirichlet_partition
from tdl.federated.training import run_fedavg_round
from tdl.models import FashionMNISTCNN
from tdl.seed import set_seed
from tdl.training.centralized import resolve_device
from tdl.training.evaluate import evaluate


def run(config_path: str | Path, output: str | Path,
        alpha: float | None = None) -> dict[str, Any]:
    """Run all-client FedAvg and save metadata, partition summary, and metrics."""
    config = load_config(config_path)
    if alpha is not None:
        config["alpha"] = alpha
    config.setdefault("min_samples_per_client", 1)
    config.setdefault("num_workers", 0)
    for key, minimum in (("rounds", 1), ("local_epochs", 1), ("batch_size", 1),
                         ("num_workers", 0), ("min_samples_per_client", 1)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    train_data, test_data = load_fashion_mnist(config["data_dir"])
    labels = train_data.targets.numpy()
    partition = create_dirichlet_partition(
        labels, config["num_clients"], config["alpha"], config["seed"],
        config["min_samples_per_client"],
    )
    partition_summary = summarize_partition(labels, partition)
    if not partition_summary["coverage_complete"]:
        raise RuntimeError("Training partition does not cover every index exactly once.")
    loader_options = {"batch_size": config["batch_size"], "num_workers": config["num_workers"],
                      "pin_memory": device.type == "cuda"}
    client_loaders = {
        client: DataLoader(Subset(train_data, indices), shuffle=True,
                           generator=torch.Generator().manual_seed(config["seed"] + client),
                           **loader_options)
        for client, indices in partition.items()
    }
    test_loader = DataLoader(test_data, shuffle=False,
                             generator=torch.Generator().manual_seed(config["seed"]), **loader_options)
    global_model = FashionMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    initial_loss, initial_accuracy = evaluate(global_model, test_loader, criterion, device)
    report = {
        "metadata": {"config": config, "device": str(device), "dataset": "Fashion-MNIST",
                     "training_samples": len(train_data), "test_samples": len(test_data),
                     "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
                     "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "client_seed_rule": "seed + round_number * num_clients + client_id (rounds start at 1)",
                     "optimizer_state": "fresh per client per round", "participation": "all clients"},
        "partition": partition_summary,
        "initial_evaluation": {"global_test_loss": initial_loss, "global_test_accuracy": initial_accuracy},
        "rounds": [],
    }
    print(f"Device: {device} | alpha: {config['alpha']} | seed: {config['seed']}")
    print(f"Initial | test loss: {initial_loss:.4f} | test accuracy: {initial_accuracy:.2%}")
    for round_number in range(1, config["rounds"] + 1):
        metrics = run_fedavg_round(global_model, client_loaders, config, device, round_number)
        loss, accuracy = evaluate(global_model, test_loader, criterion, device)
        metrics.update({"global_test_loss": loss, "global_test_accuracy": accuracy})
        report["rounds"].append(metrics)
        print(f"Round {round_number}/{config['rounds']} | "
              f"weighted client loss: {metrics['weighted_mean_client_training_loss']:.4f} | "
              f"test loss: {loss:.4f} | test accuracy: {accuracy:.2%} | "
              f"clients: {metrics['num_clients']} | samples: {metrics['total_samples']}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report saved to {output_path}")
    return report


def main() -> None:
    """Expose a config-based runner with optional alpha and report overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/fedavg_summary.json"))
    args = parser.parse_args()
    run(args.config, args.output, args.alpha)


if __name__ == "__main__":
    main()
