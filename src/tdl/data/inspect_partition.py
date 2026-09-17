"""Inspect training-label partitions without constructing or training a model."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from torchvision import datasets

from tdl.config import load_config
from tdl.data.partition import create_dirichlet_partition


def summarize_partition(labels: ArrayLike, partition: dict[int, list[int]]) -> dict[str, Any]:
    """Report class counts, exact coverage, and mean distance to global proportions.

    Total variation is half the sum of absolute label-proportion differences
    from the full dataset (0 = identical, 1 = disjoint). Empty clients receive
    zero proportions and no distance; the mean includes only nonempty clients.
    """
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array.")
    if not np.issubdtype(label_array.dtype, np.integer):
        raise ValueError("labels must contain integer class IDs.")
    classes, global_counts = np.unique(label_array, return_counts=True)
    global_proportions = global_counts / len(label_array)
    assigned = []
    clients = []
    distances = []
    for client, indices in sorted(partition.items()):
        if any(isinstance(index, bool) or not isinstance(index, (int, np.integer))
               or index < 0 or index >= len(label_array) for index in indices):
            raise ValueError("Partition indices must be integers within the dataset range.")
        assigned.extend(indices)
        client_labels = label_array[np.asarray(indices, dtype=int)]
        counts = np.array([np.count_nonzero(client_labels == label) for label in classes])
        proportions = counts / len(indices) if indices else np.zeros(len(classes))
        distance = float(np.abs(proportions - global_proportions).sum() / 2) if indices else None
        if distance is not None:
            distances.append(distance)
        clients.append({
            "client_id": client,
            "num_samples": len(indices),
            "class_counts": {str(label): int(count) for label, count in zip(classes, counts)},
            "class_proportions": {str(label): float(p) for label, p in zip(classes, proportions)},
            "total_variation_from_global": distance,
        })
    return {
        "dataset_size": len(label_array),
        "num_clients": len(partition),
        "total_assigned_samples": len(assigned),
        "coverage_complete": sorted(assigned) == list(range(len(label_array))),
        "mean_total_variation_from_global": float(np.mean(distances)) if distances else None,
        "clients": clients,
    }


def run(config_path: str | Path, output: str | Path, alpha: float | None = None) -> dict[str, Any]:
    """Load only Fashion-MNIST training labels, verify a partition, and save JSON."""
    config = load_config(config_path)
    options = {
        "num_clients": config["num_clients"],
        "alpha": config["alpha"] if alpha is None else alpha,
        "seed": config["seed"],
        "min_samples_per_client": config.get("min_samples_per_client", 1),
    }
    dataset = datasets.FashionMNIST(root=str(config["data_dir"]), train=True, download=True)
    labels = dataset.targets.numpy()
    partition = create_dirichlet_partition(labels, **options)
    repeated = create_dirichlet_partition(labels, **options)
    summary = summarize_partition(labels, partition)
    minimum_met = all(len(indices) >= options["min_samples_per_client"] for indices in partition.values())
    reproducible = partition == repeated
    if not summary["coverage_complete"] or not minimum_met or not reproducible:
        raise RuntimeError("Partition failed coverage, minimum-size, or reproducibility verification.")
    summary.update(options)
    summary.update({"dataset": "Fashion-MNIST", "split": "train",
                    "minimum_samples_met": minimum_met, "reproducibility_verified": reproducible})
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"alpha={options['alpha']} | clients={options['num_clients']} | seed={options['seed']}")
    print(f"Assigned: {summary['total_assigned_samples']}/{summary['dataset_size']} | "
          f"complete coverage: {summary['coverage_complete']} | minimum met: {minimum_met} | "
          f"reproducible: {reproducible}")
    for client in summary["clients"]:
        print(f"Client {client['client_id']}: {client['num_samples']} samples | "
              f"class counts: {client['class_counts']}")
    print(f"Mean label total variation from global: {summary['mean_total_variation_from_global']:.4f}")
    print(f"Summary saved to {output_path}")
    return summary


def main() -> None:
    """Parse inspection options; alpha overrides the YAML for development checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/partition_summary.json"))
    args = parser.parse_args()
    run(args.config, args.output, args.alpha)


if __name__ == "__main__":
    main()
