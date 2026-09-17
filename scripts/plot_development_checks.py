"""Rebuild documentation figures from a compact, source-verified snapshot.

Optional --results-dir refreshes the snapshot from the original local JSONs.
No training is performed. Install the project's [plots] extra to run this.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from tdl.decentralized.topology import TOPOLOGIES, build_topology, topology_summary


ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#0072B2", "#D55E00", "#7A5195"]
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
           "Shirt", "Sneaker", "Bag", "Ankle boot"]
TOPOLOGY_LABELS = ["Ring d2", "Ring d4", "Random Regular d4", "Small-World d4", "Fully Connected"]
TOPOLOGY_COLORS = ["#0072B2", "#D55E00", "#009E73", "#7A5195", "#333333"]


def milestone4_snapshot(results_dir, source_commit, verification_date):
    summary, source = read_report(results_dir / "milestone4_development_summary.json")
    rows = summary["rows"]
    reports = []
    for row in rows:
        report, provenance = read_report(results_dir / row["report"])
        config = report["config"]
        graph = build_topology(config["topology"], config["num_clients"], config["seed"],
                               config["small_world_rewire_probability"])
        if topology_summary(graph) != report["topology"]:
            raise ValueError("Recorded graph differs from the experiment graph generator.")
        final = report["rounds"][-1]
        for short, full in {"mean_accuracy": "mean_node_test_accuracy",
                            "worst_accuracy": "worst_node_test_accuracy",
                            "accuracy_std": "node_test_accuracy_std",
                            "mean_test_loss": "mean_node_test_loss",
                            "disagreement": "mean_pairwise_rms_parameter_distance"}.items():
            if row[short] != final[full]:
                raise ValueError("Consolidated summary differs from its raw report.")
        reports.append({"alpha": row["alpha"], "topology_name": row["topology"],
                        "source": provenance, "metadata": report["metadata"],
                        "config": config, "topology": report["topology"]})
    for check in summary["fully_connected_fedavg_sanity_checks"]:
        fedavg, _ = read_report(results_dir / check["fedavg_reference"])
        fc, _ = read_report(results_dir / f"decentralized_fully_connected_alpha_{check['alpha']:g}.json")
        if fc["partition"] != fedavg["partition"]:
            raise ValueError("Fully connected and FedAvg partitions differ.")
        for local, reference in zip(fc["rounds"], fedavg["rounds"], strict=True):
            for a, b in [("mean_node_test_accuracy", "global_test_accuracy"),
                         ("mean_node_test_loss", "global_test_loss"),
                         ("weighted_mean_local_training_loss", "weighted_mean_client_training_loss")]:
                if local[a] != reference[b]:
                    raise ValueError("Fully connected/FedAvg equality check failed.")
    code_paths = [ROOT / "configs/decentralized.yaml", ROOT / "scripts/run_decentralized_checks.py",
                  ROOT / "pyproject.toml", ROOT / "src/tdl/seed.py", ROOT / "src/tdl/models/cnn.py",
                  ROOT / "src/tdl/data/partition.py", ROOT / "src/tdl/data/fashion_mnist.py",
                  ROOT / "src/tdl/federated/training.py", ROOT / "src/tdl/federated/aggregation.py",
                  ROOT / "src/tdl/training/centralized.py", ROOT / "src/tdl/training/evaluate.py",
                  *sorted((ROOT / "src/tdl/decentralized").glob("*.py"))]
    committed_files = []
    for path in code_paths:
        name = path.relative_to(ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=ROOT)
        if path.read_bytes().replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
            raise ValueError(f"Local source differs from the requested implementation commit: {name}")
        committed_files.append({"file": name, "sha256": hashlib.sha256(committed).hexdigest()})
    return {"verification_date": verification_date,
            "source_code": {"commit": source_commit, "hash_basis": "committed Git blob bytes",
                            "relationship_to_runs": "Development runs preceded this commit; shared helpers were subsequently extracted without changing static training mathematics.",
                            "files": committed_files},
            "summary_source": source, "rows": rows, "reports": reports,
            "fully_connected_fedavg_sanity_checks": summary["fully_connected_fedavg_sanity_checks"],
            "interpretation": summary["interpretation"], "accuracy_units": summary["accuracy_units"]}


def read_report(path):
    raw = path.read_bytes()
    return json.loads(raw), {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def make_snapshot(results_dir, source_commit, verification_date, milestone4_source_commit):
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
    return {"schema_version": 2, "verification_date": verification_date,
            "source_code_commit": source_commit, "centralized_baseline": centralized,
            "interpretation": "Single-seed development validation; not final scientific results.",
            "repeat_verification": {"alpha": 0.3, "reports_identical": True, "source": repeated_source},
            "checks": checks,
            "milestone4": milestone4_snapshot(results_dir, milestone4_source_commit, verification_date)}


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
    plot_milestone4(snapshot["milestone4"], output_dir)


def plot_milestone4(data, output_dir):
    rows = data["rows"]
    expected = {(alpha, name) for alpha in [10.0, 1.0, 0.3] for name in TOPOLOGIES}
    if len(rows) != 15 or {(r["alpha"], r["topology"]) for r in rows} != expected:
        raise ValueError("Expected all fifteen Milestone 4 conditions exactly once.")
    lookup = {(r["alpha"], r["topology"]): r for r in rows}
    graphs = {}
    if len(data["reports"]) != 15:
        raise ValueError("Expected fifteen source report records.")
    reference = None
    for report in data["reports"]:
        config = report["config"]
        if any(config[k] != v for k, v in {"seed": 42, "num_clients": 10, "rounds": 3, "local_epochs": 1}.items()):
            raise ValueError("Unexpected Milestone 4 development configuration.")
        comparable = {k: v for k, v in config.items() if k not in {"alpha", "topology"}}
        reference = comparable if reference is None else reference
        if comparable != reference or report["metadata"]["training_samples"] != 60000 or report["metadata"]["test_samples"] != 10000:
            raise ValueError("Development comparisons must use the same full-data configuration.")
        graph = build_topology(report["topology_name"], 10, 42, config["small_world_rewire_probability"])
        if topology_summary(graph) != report["topology"]:
            raise ValueError("Snapshot graph differs from the experiment graph.")
        row = lookup[(report["alpha"], report["topology_name"])]
        metadata = report["topology"]
        if row["round"] != 3 or config["alpha"] != row["alpha"] or config["topology"] != row["topology"]:
            raise ValueError("Snapshot condition differs from its configuration.")
        for short, full in {"edges": "edges", "average_degree": "average_degree",
                            "minimum_degree": "minimum_degree", "maximum_degree": "maximum_degree",
                            "transmissions_per_round": "directed_model_transmissions_per_round",
                            "average_aggregation_size": "average_aggregation_size_including_self"}.items():
            if row[short] != metadata[full]:
                raise ValueError("Communication row differs from graph metadata.")
        if not all(np.isfinite(row[k]) for k in ["mean_accuracy", "worst_accuracy", "accuracy_std", "mean_test_loss", "disagreement"]):
            raise ValueError("Figure metrics must be finite.")
        graphs[report["topology_name"]] = graph
    caveat = "Seed 42 only | 3 rounds | development checks | no confidence intervals or significance claims"
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    positions = nx.circular_layout(range(10))
    for ax, name, label, color in zip(axes.flat, TOPOLOGIES, TOPOLOGY_LABELS, TOPOLOGY_COLORS):
        row = lookup[(10.0, name)]
        nx.draw_networkx(graphs[name], pos=positions, ax=ax, node_color=color, node_size=370,
                         font_color="white", font_size=10, edge_color="#718096", width=1.1)
        ax.set_title(f"{label}\n{row['edges']} edges | mean degree {row['average_degree']:g}\n"
                     f"{row['transmissions_per_round']} transmissions / round", fontsize=11)
        ax.margins(0.15)
        ax.set_axis_off()
    axes.flat[-1].set_axis_off()
    axes.flat[-1].text(0.05, 0.65, "Nodes: simulated learners 0–9\nEdges: direct model exchange\nSelf included in aggregation; no self-loop\nTraffic excludes self\nRandom graphs: seed 42\nSmall-world rewiring: p = 0.2", fontsize=11, va="top")
    fig.suptitle("Decentralized communication topologies\n"
                 "10 simulated nodes | seeded graph construction | development configuration", fontsize=13)
    save(fig, output_dir, "decentralized_topologies")

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for name, label, color, marker in zip(TOPOLOGIES, TOPOLOGY_LABELS, TOPOLOGY_COLORS, ["o", "s", "^", "D", "X"]):
        ax.plot(range(3), [100 * lookup[(alpha, name)]["mean_accuracy"] for alpha in [10.0, 1.0, 0.3]],
                color=color, marker=marker, markersize=7, label=label)
    ax.set_xticks(range(3), ["Alpha 10", "Alpha 1", "Alpha 0.3"])
    ax.set(xlabel="Lower alpha → stronger label skew (three observed conditions only)",
           ylabel="Round-3 mean node test accuracy (%)", ylim=(63, 87))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", fontsize=10)
    fig.suptitle("Heterogeneity and decentralized accuracy\n" + caveat, fontsize=12)
    save(fig, output_dir, "decentralized_heterogeneity_accuracy")

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    offsets = [(12, -28), (25, -58), (25, -10), (25, 32), (-145, 20)]
    for name, label, color, offset in zip(TOPOLOGIES, TOPOLOGY_LABELS, TOPOLOGY_COLORS, offsets):
        row = lookup[(0.3, name)]
        x, y = row["transmissions_per_round"], 100 * row["mean_accuracy"]
        ax.scatter(x, y, color=color, s=85, zorder=3)
        disagreement = "0" if row["disagreement"] == 0 else f"{row['disagreement']:.2e}"
        ax.annotate(f"{label}\nRMS disagreement = {disagreement}", (x, y),
                    xytext=offset, textcoords="offset points", fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": color}, color=color)
    ax.set(xlabel="Directed full-model transmissions per round (idealized; excludes self)",
           ylabel="Round-3 mean node test accuracy (%)", xlim=(10, 103), ylim=(64, 81))
    ax.set_xticks([20, 40, 90])
    ax.grid(alpha=0.2)
    fig.suptitle("Communication and accuracy | alpha = 0.3\n" + caveat, fontsize=12)
    save(fig, output_dir, "decentralized_communication_accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "docs/figure_data/development_checks.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--results-dir", type=Path, help="Refresh the snapshot from original local reports")
    parser.add_argument("--source-commit", help="Experiment source commit, required when refreshing")
    parser.add_argument("--verification-date", help="Experiment verification date, required when refreshing")
    parser.add_argument("--milestone4-source-commit", help="Committed Milestone 4 implementation revision, required when refreshing")
    args = parser.parse_args()
    if args.results_dir:
        if not args.source_commit or not args.verification_date or not args.milestone4_source_commit:
            parser.error("--results-dir requires --source-commit, --milestone4-source-commit and --verification-date")
        previous = json.loads(args.snapshot.read_text(encoding="utf-8")) if args.snapshot.exists() else None
        snapshot = make_snapshot(args.results_dir, args.source_commit, args.verification_date, args.milestone4_source_commit)
        if previous and "milestone4" in previous:
            old = previous["milestone4"]
            snapshot["milestone4"]["original_run_source_files"] = old.get("original_run_source_files", old["source_code"]["files"])
        validate_snapshot(snapshot)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    plot_figures(snapshot, args.output_dir)
    print(f"Validated snapshot and generated eight PNG/SVG figure pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
