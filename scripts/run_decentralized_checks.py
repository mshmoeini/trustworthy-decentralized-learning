"""Run modest local topology checks and summarize ignored JSON/CSV reports.

Existing reports are never overwritten: explicit requested checks must be new.
The summary includes every matching decentralized topology/alpha report found.
"""

import argparse
import csv
import json
from pathlib import Path
import time

from tdl.decentralized.runner import run
from tdl.decentralized.topology import TOPOLOGIES


def summarize(results_dir: Path) -> dict:
    rows, sanity = [], []
    for path in sorted(results_dir.glob("decentralized_*_alpha_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report["config"]
        final = report["rounds"][-1]
        graph = report["topology"]
        row = {"topology": config["topology"], "alpha": config["alpha"], "seed": config["seed"],
               "round": final["round"], "device": report["metadata"]["device"],
               "mean_accuracy": final["mean_node_test_accuracy"],
               "worst_accuracy": final["worst_node_test_accuracy"],
               "accuracy_std": final["node_test_accuracy_std"],
               "mean_test_loss": final["mean_node_test_loss"],
               "disagreement": final["mean_pairwise_rms_parameter_distance"],
               "edges": graph["edges"], "average_degree": graph["average_degree"],
               "minimum_degree": graph["minimum_degree"], "maximum_degree": graph["maximum_degree"],
               "transmissions_per_round": graph["directed_model_transmissions_per_round"],
               "average_aggregation_size": graph["average_aggregation_size_including_self"],
               "report": path.name}
        rows.append(row)
        if config["topology"] == "fully_connected":
            suffix = f"{config['alpha']:g}"
            reference_path = results_dir / f"fedavg_alpha_{suffix}.json"
            # Existing FedAvg uses 0.3, 1, 10 filename suffixes.
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            reference_config = reference["metadata"]["config"]
            matching = all(config[key] == value for key, value in reference_config.items())
            identical_partition = report["partition"] == reference["partition"]
            if not matching or not identical_partition:
                raise ValueError("Fully connected/FedAvg sanity check requires identical training configs and partitions.")
            differences = []
            for decentralized, fedavg in zip(report["rounds"], reference["rounds"], strict=True):
                differences.append({"round": decentralized["round"],
                    "accuracy_absolute_difference": abs(decentralized["mean_node_test_accuracy"] - fedavg["global_test_accuracy"]),
                    "test_loss_absolute_difference": abs(decentralized["mean_node_test_loss"] - fedavg["global_test_loss"]),
                    "training_loss_absolute_difference": abs(decentralized["weighted_mean_local_training_loss"]
                                                             - fedavg["weighted_mean_client_training_loss"]),
                    "node_accuracy_std": decentralized["node_test_accuracy_std"],
                    "node_disagreement": decentralized["mean_pairwise_rms_parameter_distance"]})
            sanity.append({"alpha": config["alpha"], "same_training_config": matching,
                           "identical_partition": identical_partition, "fedavg_reference": reference_path.name,
                           "rounds": differences})
    rows.sort(key=lambda r: (-r["alpha"], TOPOLOGIES.index(r["topology"])))
    summary = {"interpretation": "Single-seed local development checks, not final scientific results.",
               "accuracy_units": "fractions; std is population std across nodes",
               "rows": rows, "fully_connected_fedavg_sanity_checks": sanity}
    (results_dir / "milestone4_development_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (results_dir / "milestone4_development_summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Topology | alpha | mean acc | worst acc | std | mean loss | disagreement | edges | tx/round")
    for row in rows:
        print(f"{row['topology']} | {row['alpha']:g} | {row['mean_accuracy']:.2%} | "
              f"{row['worst_accuracy']:.2%} | {row['accuracy_std']:.2%} | {row['mean_test_loss']:.4f} | "
              f"{row['disagreement']:.8f} | {row['edges']} | {row['transmissions_per_round']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/decentralized.yaml"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.3])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        for alpha in args.alphas:
            for topology in TOPOLOGIES:
                path = args.results_dir / f"decentralized_{topology}_alpha_{alpha:g}.json"
                if path.exists():
                    raise FileExistsError(f"Preserving existing report: {path}; use --summary-only to inspect it.")
                started = time.perf_counter()
                run(args.config, path, alpha=alpha, topology=topology)
                seconds = time.perf_counter() - started
                print(f"Completed {topology} alpha={alpha:g} in {seconds:.1f} s", flush=True)
    summarize(args.results_dir)


if __name__ == "__main__":
    main()
