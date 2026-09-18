"""Archive compact verified Morph aggregates, or regenerate the public figure.

Normal plotting requires only the versioned snapshot; no training or raw reports.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ["Morph code-faithful", "EL-Local uniform"]
METRICS = {"mean_accuracy": "mean_node_test_accuracy", "worst_accuracy": "worst_node_test_accuracy",
           "accuracy_std": "node_test_accuracy_std", "mean_test_loss": "mean_node_test_loss",
           "disagreement": "mean_pairwise_rms_parameter_distance"}


def archive(snapshot, results):
    path = results / "morph_code_faithful_vs_el_alpha_0.3_multiseed_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    records = []
    source_groups = []
    values = {algorithm: {metric: [] for metric in METRICS} for algorithm in ALGORITHMS}
    reports = {}
    for provenance in summary["provenance"]:
        algorithm, seed = provenance["algorithm"], provenance["seed"]
        if algorithm not in ALGORITHMS or seed not in range(42, 47) or (algorithm, seed) in reports:
            raise ValueError("Unexpected or duplicate paired run")
        report_path = results / provenance["file"]
        raw = report_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != provenance["sha256"]:
            raise ValueError("Report differs from validated summary")
        report = json.loads(raw)
        config, rounds, metadata = report["config"], report["rounds"], report["metadata"]
        if any(config[k] != v for k, v in {"seed": seed, "alpha": .3, "rounds": 10, "num_clients": 10, "local_epochs": 1}.items()):
            raise ValueError("Unexpected validation settings")
        if [r["round"] for r in rounds] != list(range(1, 11)):
            raise ValueError("Incomplete ten-round report")
        for r in rounds:
            topology = r["topology"]
            if r["directed_model_transmissions"] != 40 or sum(topology["in_degrees"].values()) != 40 or sum(topology["out_degrees"].values()) != 40:
                raise ValueError("Measured model budget mismatch")
            if algorithm == ALGORITHMS[0] and (set(topology["in_degrees"].values()) != {4} or r["declined_requests"] != 0):
                raise ValueError("Primary serving invariant mismatch")
        if algorithm == ALGORITHMS[0]:
            if config["morph_mode"] != "morph_code_faithful" or config["outgoing_capacity"] is not None:
                raise ValueError("Expected primary uncapped mode")
            for k, v in {"k": 4, "beta": 500., "topology_refresh_interval": 1, "random_peer_count": 1, "selection_mode": "paper_resample"}.items():
                if config[k] != v:
                    raise ValueError("Primary selection settings differ")
        elif config["epidemic_k"] != 4 or config["aggregation_mode"] != "paper_epidemic":
            raise ValueError("Expected uniform EL control")
        row = next(r for r in summary["rows"] if r["algorithm"] == algorithm and r["seed"] == seed)
        for short, full in METRICS.items():
            value = rounds[-1][full]
            if not math.isfinite(value) or row[short] != value:
                raise ValueError("Final metric mismatch")
            values[algorithm][short].append(value)
        implementation = metadata["local_implementation"]
        hashes = implementation["source_sha256"]
        if hashes not in source_groups:
            source_groups.append(hashes)
        records.append({"algorithm": algorithm, "seed": seed, "file": report_path.name,
                        "sha256": provenance["sha256"], "config_sha256": implementation["config_sha256"],
                        "base_commit": implementation["git_head"], "runtime_source_group": source_groups.index(hashes),
                        "reused_existing_report": provenance["reused_existing_report"]})
        reports[(algorithm, seed)] = report
    if len(reports) != 10:
        raise ValueError("Expected five complete paired seeds")
    for seed in range(42, 47):
        morph, el = [reports[(a, seed)] for a in ALGORITHMS]
        initial_m, initial_e = [r["initial_node_evaluation"] for r in [morph, el]]
        initial_nodes = lambda evaluation: [{k: node[k] for k in ["node_id", "test_loss", "test_accuracy"]} for node in evaluation["nodes"]]
        if (morph["partition"] != el["partition"] or initial_nodes(initial_m) != initial_nodes(initial_e)
                or any(initial_m[k] != initial_e[k] for k in METRICS.values())):
            raise ValueError("Paired data or initialization differs")
        for key in ["seed", "num_clients", "alpha", "rounds", "local_epochs", "batch_size", "learning_rate", "momentum", "optimizer", "num_workers", "min_samples_per_client"]:
            if morph["config"][key] != el["config"][key]:
                raise ValueError("Paired learning controls differ")
    aggregates = {}
    for algorithm, metrics in values.items():
        aggregates[algorithm] = {}
        for metric, samples in metrics.items():
            computed = {"mean": statistics.mean(samples), "std": statistics.stdev(samples),
                        "min": min(samples), "max": max(samples), "n": 5}
            if any(computed[k] != summary["across_seed"][algorithm][metric][k] for k in computed):
                raise ValueError("Aggregate mismatch")
            aggregates[algorithm][metric] = computed
    config = reports[(ALGORITHMS[0], 42)]["config"]
    snapshot["morph_multiseed"] = {
        "verification_date": "2026-09-18", "dataset": "Fashion-MNIST", "alpha": .3,
        "seeds": list(range(42, 47)), "num_clients": 10, "rounds": 10, "local_epochs": 1,
        "model_deliveries_per_round": 40, "total_model_deliveries_per_run": 400,
        "aggregation": "uniform self plus incoming", "primary_mode": "morph_code_faithful",
        "selection": {k: config[k] for k in ["k", "beta", "topology_refresh_interval", "random_peer_count", "selection_mode"]},
        "learning_controls": {k: config[k] for k in ["batch_size", "learning_rate", "momentum", "optimizer", "num_workers", "min_samples_per_client"]},
        "source_revision": "57d73b921c317b0f0d7d9f4a71a6db8051beaf82",
        "std_definition": "Across-seed sample standard deviation, denominator n-1; not confidence intervals",
        "accuracy_units": "Fractions; multiply by 100 for percent and percentage-point node std",
        "interpretation": "Development validation only; descriptive results, no significance claims. Code-faithful serving with requested paper-hybrid selection adaptation.",
        "aggregates": aggregates,
        "summary_source": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "report_sources": sorted(records, key=lambda r: (r["seed"], r["algorithm"])),
        "runtime_source_groups": source_groups,
        "provenance_note": "Runtime source hashes identify uncommitted experiment bytes; base_commit is not the Morph implementation commit. Existing EL42 report reused; its historical Morph hash belongs only to the unchanged observational cosine helper. Raw reports remain ignored."
    }


def plot(snapshot, output):
    data = snapshot["morph_multiseed"]
    if data["seeds"] != list(range(42, 47)) or data["rounds"] != 10 or data["model_deliveries_per_round"] != 40:
        raise ValueError("Unexpected published figure settings")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "tdl-morph-multiseed",
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8))
        colors = ["#0072B2", "#D55E00"]
        for ax, metric, title, limits in zip(axes,
                ["mean_accuracy", "worst_accuracy", "accuracy_std"],
                ["Mean accuracy", "Worst-node accuracy", "Node accuracy std"],
                [(0, 100), (0, 100), (0, 5)]):
            aggregates = [data["aggregates"][a][metric] for a in ALGORITHMS]
            means = [100 * s["mean"] for s in aggregates]
            errors = [100 * s["std"] for s in aggregates]
            if any(not math.isfinite(v) or v < 0 for v in means + errors):
                raise ValueError("Invalid plotted aggregate")
            ax.bar(range(2), means, yerr=errors, color=colors, width=.62,
                   capsize=6, error_kw={"elinewidth": 1.4, "capthick": 1.4})
            for i, mean, error in zip(range(2), means, errors):
                ax.text(i, mean + error + (2 if metric != "accuracy_std" else .13), f"{mean:.2f}", ha="center")
            ax.set_xticks(range(2), ["Morph", "EL-Local"])
            ax.set_title(title)
            ax.set_ylabel("Percentage points" if metric == "accuracy_std" else "Accuracy (%)")
            ax.set_ylim(*limits)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
        fig.suptitle("Morph vs EL-Local | development validation\n"
                     "5 seeds | alpha = 0.3 | 10 rounds | 10 nodes | 40 model deliveries / round", fontsize=12, y=.96)
        fig.text(.5, .035, "Final-round means; error bars: across-seed sample standard deviation (n = 5). No significance claim.", ha="center", fontsize=9)
        fig.subplots_adjust(left=.065, right=.985, bottom=.16, top=.74, wspace=.42)
        output.mkdir(parents=True, exist_ok=True)
        for extension in ["png", "svg"]:
            path = output / f"morph_vs_epidemic_multiseed.{extension}"
            fig.savefig(path, dpi=180,
                        metadata={"Date": None} if extension == "svg" else None)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "docs/figure_data/development_checks.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--archive-results", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if args.archive_results:
        archive(snapshot, args.archive_results)
        args.snapshot.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot(snapshot, args.output_dir)
    print(f"Generated Morph PNG/SVG pair in {args.output_dir}")


if __name__ == "__main__":
    main()
