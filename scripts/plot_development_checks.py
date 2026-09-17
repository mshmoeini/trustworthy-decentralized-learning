"""Rebuild documentation figures from a compact, source-verified snapshot.

Optional --results-dir refreshes the snapshot from the original local JSONs.
No training is performed. Install the project's [plots] extra to run this.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#0072B2", "#D55E00", "#7A5195"]
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
           "Shirt", "Sneaker", "Bag", "Ankle boot"]


def read_report(path):
    raw = path.read_bytes()
    return json.loads(raw), {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def make_snapshot(results_dir, source_commit, verification_date):
    checks = []
    for alpha, suffix in [(10.0, "10"), (1.0, "1"), (0.3, "0.3")]:
        fedavg, fedavg_source = read_report(results_dir / f"fedavg_alpha_{suffix}.json")
        partition, partition_source = read_report(results_dir / f"partition_alpha_{suffix}.json")
        clients = [{"client_id": c["client_id"], "num_samples": c["num_samples"],
                    "class_counts": c["class_counts"]} for c in partition["clients"]]
        training_clients = [{key: c[key] for key in clients[0]} for c in fedavg["partition"]["clients"]]
        if clients != training_clients or not partition["coverage_complete"]:
            raise ValueError("Partition inspection and FedAvg partitions must agree and cover the dataset.")
        if partition["alpha"] != alpha or fedavg["metadata"]["config"]["alpha"] != alpha:
            raise ValueError("Source alpha differs from the expected check.")
        keys = ["round", "weighted_mean_client_training_loss", "global_test_loss",
                "global_test_accuracy", "num_clients", "total_samples"]
        checks.append({"alpha": alpha, "metadata": fedavg["metadata"],
                       "sources": [partition_source, fedavg_source], "clients": clients,
                       "initial_evaluation": fedavg["initial_evaluation"],
                       "rounds": [{key: r[key] for key in keys} for r in fedavg["rounds"]]})
    repeated, repeated_source = read_report(results_dir / "fedavg_alpha_0.3_repeat.json")
    original, _ = read_report(results_dir / "fedavg_alpha_0.3.json")
    if repeated != original:
        raise ValueError("The repeated alpha 0.3 run differs from the original.")
    log_path = ROOT / "docs/research_log.md"
    log_raw = log_path.read_bytes()
    historical_section = log_raw.decode("utf-8").split("## CUDA development environment verification")[0]
    rows = re.findall(r"\| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)% \|", historical_section)
    if [int(row[0]) for row in rows] != [1, 2, 3]:
        raise ValueError("Expected the three recorded centralized development epochs.")
    centralized = {
        "device": "cpu", "verification_date": "2026-09-14", "seed": 42,
        "source": {"file": "docs/research_log.md", "section": "Milestone 1: centralized development baseline",
                   "sha256": hashlib.sha256(log_raw).hexdigest()},
        "precision": "Recorded losses rounded to 4 decimals; accuracy rounded to 2 percentage decimals.",
        "epochs": [{"epoch": int(epoch), "training_loss": float(train), "test_loss": float(test),
                    "test_accuracy": float(accuracy) / 100} for epoch, train, test, accuracy in rows],
    }
    return {"schema_version": 1, "verification_date": verification_date,
            "source_code_commit": source_commit, "centralized_baseline": centralized,
            "interpretation": "Single-seed development validation; not final scientific results.",
            "repeat_verification": {"alpha": 0.3, "reports_identical": True, "source": repeated_source},
            "checks": checks}


def validate_snapshot(snapshot):
    checks = snapshot["checks"]
    if [c["alpha"] for c in checks] != [10.0, 1.0, 0.3]:
        raise ValueError("Expected alpha checks 10, 1, and 0.3 in that order.")
    reference = {k: v for k, v in checks[0]["metadata"]["config"].items() if k != "alpha"}
    for check in checks:
        config = check["metadata"]["config"]
        if any(config[key] != value for key, value in
               {"seed": 42, "num_clients": 10, "local_epochs": 1, "rounds": 3}.items()):
            raise ValueError("These documentation plots describe the recorded seed-42 development checks.")
        if {k: v for k, v in config.items() if k != "alpha"} != reference:
            raise ValueError("Alpha comparisons must have identical other configurations.")
        counts = count_matrix(check)
        sizes = np.array([c["num_samples"] for c in check["clients"]])
        if (counts.shape != (10, 10) or np.any(counts < 0)
                or not np.array_equal(counts.sum(axis=1), sizes)
                or not np.all(counts.sum(axis=0) == 6000)
                or np.any(sizes < config["min_samples_per_client"])):
            raise ValueError("Invalid full Fashion-MNIST class counts or client sizes.")
        if [r["round"] for r in check["rounds"]] != list(range(1, config["rounds"] + 1)):
            raise ValueError("Round records are missing or unordered.")
        if any(r["num_clients"] != 10 or r["total_samples"] != 60000 for r in check["rounds"]):
            raise ValueError("All clients and samples must participate in each round.")


def count_matrix(check):
    return np.array([[c["class_counts"][str(label)] for label in range(10)]
                     for c in check["clients"]], dtype=int)


def save(fig, output_dir, name):
    for extension in ["png", "svg"]:
        fig.savefig(output_dir / f"{name}.{extension}", dpi=200,
                    metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            svg_path = output_dir / f"{name}.svg"
            svg_path.write_text("\n".join(line.rstrip() for line in
                                         svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                                encoding="utf-8")
    plt.close(fig)


def plot_figures(snapshot, output_dir):
    validate_snapshot(snapshot)
    checks = snapshot["checks"]
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "tdl-development-checks"})
    subtitle = "Fashion-MNIST training set | 10 clients | seed 42 | single-seed development check"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), layout="constrained")
    for ax, check in zip(axes, checks):
        counts = count_matrix(check)
        image = ax.imshow(counts / counts.sum(axis=1, keepdims=True), vmin=0, vmax=1,
                          cmap="viridis", aspect="auto", interpolation="nearest")
        ax.set(title=f"Alpha = {check['alpha']:g}", ylabel="Client ID", xlabel="Class")
        ax.set_yticks(range(10))
        ax.set_xticks(range(10), CLASSES, rotation=55, ha="right")
    fig.colorbar(image, ax=axes, label="Within-client class proportion", shrink=0.8)
    fig.suptitle("Client label distributions\n" + subtitle, fontsize=13)
    save(fig, output_dir, "client_label_distributions")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, check, color in zip(axes, checks, COLORS):
        sizes = count_matrix(check).sum(axis=1)
        ax.bar(range(10), sizes, color=color)
        ax.axhline(6000, color="#555555", linestyle="--", linewidth=1, label="Equal-size reference")
        ax.set(title=f"Alpha = {check['alpha']:g}", xlabel="Client ID", ylabel="Training samples",
               ylim=(0, 12000))
        ax.set_xticks(range(10))
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Unequal client sample counts\n" + subtitle, fontsize=13)
    save(fig, output_dir, "client_sample_counts")

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for position, (check, color) in enumerate(zip(checks, COLORS)):
        counts = count_matrix(check)
        global_p = counts.sum(axis=0) / counts.sum()
        tv = np.abs(counts / counts.sum(axis=1, keepdims=True) - global_p).sum(axis=1) / 2
        ax.scatter(position + np.linspace(-0.12, 0.12, len(tv)), tv, color=color,
                   alpha=0.7, s=40)
        ax.scatter(position, tv.mean(), marker="D", s=90, color=color, edgecolor="black", zorder=3)
        ax.text(position + 0.18, tv.mean(), f"{tv.mean():.4f}", va="center", fontsize=10)
    ax.set_xticks(range(3), [f"Alpha = {c['alpha']:g}" for c in checks])
    ax.set(ylabel="Label total variation from global distribution", ylim=(0, 1), xlim=(-0.4, 2.6))
    ax.grid(axis="y", alpha=0.2)
    ax.scatter([], [], color="#555555", label="One client (horizontal offset separates points)")
    ax.scatter([], [], color="#555555", marker="D", edgecolor="black", label="Unweighted client mean")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Observed label heterogeneity\n" + subtitle, fontsize=13)
    save(fig, output_dir, "label_heterogeneity")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for check, color, marker in zip(checks, COLORS, ["o", "s", "^"]):
        rounds = check["rounds"]
        x = [0] + [r["round"] for r in rounds]
        axes[0].plot(x, [100 * check["initial_evaluation"]["global_test_accuracy"]]
                     + [100 * r["global_test_accuracy"] for r in rounds], color=color, marker=marker,
                     label=f"Alpha = {check['alpha']:g}")
        axes[1].plot(x, [check["initial_evaluation"]["global_test_loss"]]
                     + [r["global_test_loss"] for r in rounds], color=color, marker=marker)
        axes[2].plot(x[1:], [r["weighted_mean_client_training_loss"] for r in rounds],
                     color=color, marker=marker)
    for ax, title, ylabel in zip(axes, ["Global test accuracy", "Global test loss", "Local training loss"],
                                ["Accuracy (%)", "Cross-entropy", "Sample-weighted cross-entropy"]):
        ax.set(title=title, xlabel="Communication round", ylabel=ylabel, xlim=(-0.1, 3.1))
        ax.set_xticks(range(4))
        ax.grid(alpha=0.2)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    fig.suptitle("FedAvg development learning curves\n"
                 "Seed 42 | 10 clients | 60,000 training samples | 1 local epoch per round | CUDA", fontsize=13)
    save(fig, output_dir, "fedavg_learning_curves")

    baseline = snapshot["centralized_baseline"]
    records = baseline["epochs"]
    epochs = [r["epoch"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(epochs, [r["training_loss"] for r in records], color=COLORS[0], marker="o", label="Training")
    axes[0].plot(epochs, [r["test_loss"] for r in records], color=COLORS[1], marker="s", label="Test")
    axes[0].set(title="Recorded training and test loss", ylabel="Cross-entropy", ylim=(0, 0.6))
    axes[0].legend()
    axes[1].plot(epochs, [100 * r["test_accuracy"] for r in records], color=COLORS[0], marker="o")
    axes[1].set(title="Recorded test accuracy", ylabel="Accuracy (%)", ylim=(0, 100))
    for ax in axes:
        ax.set(xlabel="Centralized training epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.2)
    fig.suptitle("Historical centralized development baseline\n"
                 "CPU | seed 42 | recorded rounded metrics | 2026-09-14", fontsize=13)
    save(fig, output_dir, "centralized_baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "docs/figure_data/development_checks.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--results-dir", type=Path, help="Refresh the snapshot from original local reports")
    parser.add_argument("--source-commit", help="Experiment source commit, required when refreshing")
    parser.add_argument("--verification-date", help="Experiment verification date, required when refreshing")
    args = parser.parse_args()
    if args.results_dir:
        if not args.source_commit or not args.verification_date:
            parser.error("--results-dir requires --source-commit and --verification-date")
        snapshot = make_snapshot(args.results_dir, args.source_commit, args.verification_date)
        validate_snapshot(snapshot)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    plot_figures(snapshot, args.output_dir)
    print(f"Validated sources and generated five PNG/SVG figure pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
