"""Archive verified EL reports or plot the published snapshot without training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "22aeb9ea560a20454a7bee8e37ecc8b672867d6d"


def archive(snapshot, results):
    summary_path = results / "epidemic_development_comparison.json"
    summary = json.loads(summary_path.read_text())
    historical = {(r["topology"], r["alpha"]): r for r in snapshot["milestone4"]["rows"]}
    for row in summary["rows"]:
        if (row["algorithm"], row["alpha"]) in historical:
            old = historical[(row["algorithm"], row["alpha"])]
            for key in ["mean_accuracy", "worst_accuracy", "accuracy_std", "mean_test_loss", "disagreement", "transmissions_per_round"]:
                if row[key] != old[key]:
                    raise ValueError("Static comparison differs from historical snapshot")
    reports = []
    for path in sorted(results.glob("epidemic_*k*_alpha_*.json")):
        raw = path.read_bytes()
        report = json.loads(raw)
        rows = [r for r in summary["rows"] if r["report"] == path.name]
        if not rows:
            rows = [r for r in summary["connectivity_diagnostics"] if r["k"] == report["config"]["epidemic_k"] and r["round"] == 3]
        if len(rows) != 1 or rows[0]["mean_accuracy"] != report["rounds"][-1]["mean_node_test_accuracy"]:
            raise ValueError(f"Summary mismatch: {path}")
        for key, expected in {"seed": 42, "num_clients": 10, "rounds": 3, "local_epochs": 1}.items():
            if report["config"][key] != expected:
                raise ValueError("Unexpected EL development configuration")
        final = report["rounds"][-1]
        for short, full in [("worst_accuracy", "worst_node_test_accuracy"), ("accuracy_std", "node_test_accuracy_std"), ("disagreement", "mean_pairwise_rms_parameter_distance")]:
            if rows[0][short] != final[full]:
                raise ValueError("Summary metric mismatch")
        reports.append({"source": {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()},
                        "config": report["config"], "metadata": {k: report["metadata"][k] for k in
                            ["algorithm", "aggregation_rule", "device", "training_samples", "test_samples", "source_revision", "comparison_caveat"]},
                        "rounds": [{**{k: r[k] for k in ["round", "mean_node_test_accuracy", "worst_node_test_accuracy",
                            "node_test_accuracy_std", "mean_node_test_loss", "mean_pairwise_rms_parameter_distance", "directed_model_transmissions"]},
                            "zero_in_degree_count": r["topology"]["zero_in_degree_count"]} for r in report["rounds"]]})
    files = []
    for name in ["src/tdl/decentralized/epidemic.py", "src/tdl/decentralized/training.py", "configs/epidemic.yaml", "scripts/run_epidemic_checks.py"]:
        blob = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=ROOT)
        if (ROOT / name).read_bytes().replace(b"\r\n", b"\n") != blob.replace(b"\r\n", b"\n"):
            raise ValueError(f"Implementation changed: {name}")
        files.append({"file": name, "sha256": hashlib.sha256(blob).hexdigest()})
    snapshot["epidemic"] = {"source_code": {"commit": COMMIT, "hash_basis": "committed Git blob bytes", "files": files},
                            "summary_source": {"file": summary_path.name, "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()},
                            "rows": summary["rows"], "reports": reports, "interpretation": summary["interpretation"]}
    for report in reports:
        k = report["config"]["epidemic_k"]
        if k != 4:
            snapshot["epidemic"]["rows"].append({"algorithm": f"epidemic_local_k{k}", "alpha": .3,
                "transmissions_per_round": 10*k, "mean_accuracy": report["rounds"][-1]["mean_node_test_accuracy"]})


def plot(snapshot, output):
    from plot_development_checks import save
    rows = snapshot["epidemic"]["rows"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "tdl-epidemic", "axes.spines.top": False, "axes.spines.right": False})
    names = ["ring_degree_4", "random_regular_degree_4", "small_world_degree_4", "epidemic_local_k4", "epidemic_sample_weighted_control_k4", "fully_connected"]
    labels = ["Ring d4", "Random Regular d4", "Small-World d4", "EL-Local k4 (uniform)", "EL k4 sample-weighted control", "Fully Connected"]
    def row(name, alpha, budget):
        found = [r for r in rows if r["algorithm"] == name and r["alpha"] == alpha and r["transmissions_per_round"] == budget]
        if len(found) != 1:
            raise ValueError("Expected one condition")
        return found[0]
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for i, (name, label) in enumerate(zip(names, labels)):
        budget = 90 if name == "fully_connected" else 40
        ax.plot(range(3), [100 * row(name, a, budget)["mean_accuracy"] for a in [10, 1, .3]],
                marker=["o", "s", "^", "D", "v", "*"][i], linestyle="--" if budget == 90 else "-",
                label=f"{label} | {budget} transmissions/round")
    ax.set_xticks(range(3), ["alpha = 10", "alpha = 1", "alpha = 0.3"])
    ax.set(ylabel="Mean node test accuracy (%)", ylim=(68, 86))
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.suptitle("Approximately matched communication budget\n10 nodes | seed 42 | 3 rounds | development checks")
    save(fig, output, "epidemic_matched_budget_accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for budget, label, name in [(30, "EL k3", "epidemic_local_k3"), (40, "EL k4", "epidemic_local_k4"), (70, "EL k7", "epidemic_local_k7"), (90, "Fully Connected reference", "fully_connected")]:
        accuracy = 100 * row(name, .3, budget)["mean_accuracy"]
        ax.scatter(budget, accuracy, marker="s" if budget == 90 else "o")
        ax.annotate(label, (budget, accuracy), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set(xlabel="Model transmissions per round", ylabel="Mean node test accuracy (%)", xlim=(20, 110), ylim=(68, 80))
    ax.set_xticks([30, 40, 70, 90])
    ax.grid(alpha=.2)
    fig.suptitle("EL connectivity diagnostic | alpha = 0.3\n10 nodes | seed 42 | 3 rounds | development checks")
    save(fig, output, "epidemic_connectivity_diagnostic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-results", type=Path)
    args = parser.parse_args()
    path = ROOT / "docs/figure_data/development_checks.json"
    snapshot = json.loads(path.read_text())
    if args.archive_results:
        archive(snapshot, args.archive_results)
        path.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot(snapshot, ROOT / "docs/figures")
