"""Rebuild the two Byzantine development figures from versioned compact data.

No training, downloads, or ignored raw experiment reports are required.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def aggregate(values):
    if len(values) != 5 or any(not math.isfinite(v) for v in values):
        raise ValueError("Expected five finite seed values")
    return statistics.mean(values), statistics.stdev(values)


def save(fig, output, name):
    for extension in ("png", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=180,
                    metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in
                            path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8")
    plt.close(fig)


def plot(data, output):
    setup = data["setup"]
    if (setup["seeds"] != list(range(42, 47)) or setup["alpha"] != .3
            or setup["nodes"] != 10 or setup["rounds"] != 10
            or setup["byzantine_nodes"] != [0]):
        raise ValueError("Unexpected development validation setup")
    output.mkdir(parents=True, exist_ok=True)
    caption = "5 seeds | alpha = 0.3 | 10 nodes | 10 rounds | one Byzantine node"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "tdl-byzantine-validation",
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(9, 5))
        groups = data["honest_mean_accuracy_change_pp"]
        values = [aggregate(g["per_seed"]) for g in groups]
        means, errors = zip(*values)
        ax.bar(range(4), means, yerr=errors, width=.6, capsize=6,
               color=["#0072B2", "#0072B2", "#D55E00", "#D55E00"],
               error_kw={"elinewidth": 1.4, "capthick": 1.4})
        ax.axhline(0, color="#222222", linewidth=1.4)
        for i, mean in enumerate(means):
            ax.text(i, .65, f"{mean:+.3f} pp", ha="center", fontsize=10)
        ax.set_xticks(range(4), [g["label"].replace(" / ", "\n") for g in groups])
        ax.set_ylabel("Honest mean accuracy: attack minus clean (pp)")
        ax.set_ylim(-12, 2)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        fig.suptitle("Byzantine attack damage | development validation\n" + caption,
                     fontsize=12, y=.96)
        fig.text(.5, .035, "Final round, paired within seed. Error bars: sample standard deviation; no significance claim.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.12, right=.98, bottom=.18, top=.77)
        save(fig, output, "byzantine_attack_damage")

        fig, ax = plt.subplots(figsize=(8, 5))
        groups = data["morph_attacker_origin_delivery_share_percent"]
        values = [aggregate(g["per_seed"]) for g in groups]
        means, errors = zip(*values)
        ax.bar(range(3), means, yerr=errors, width=.55, capsize=6,
               color=["#777777", "#0072B2", "#D55E00"], alpha=.85)
        for i, (group, mean, error) in enumerate(zip(groups, means, errors)):
            ax.scatter([i + (j-2)*.055 for j in range(5)], group["per_seed"],
                       s=25, color="#222222", zorder=3)
            ax.text(i, max(max(group["per_seed"]), mean+error)+.8,
                    f"{mean:.2f}%", ha="center")
        ax.set_xticks(range(3), [g["label"] for g in groups])
        ax.set_ylabel("Attacker-origin share of model deliveries (%)")
        ax.set_ylim(0, 25)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        fig.suptitle("Morph selection amplification | development validation\n" + caption,
                     fontsize=12, y=.96)
        fig.text(.5, .055, "400 deliveries per run. Points: individual seeds; error bars: sample standard deviation.",
                 ha="center", fontsize=9)
        fig.text(.5, .018, "Clean origin deliveries are not poisoned. Selection share does not establish causation.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.12, right=.98, bottom=.16, top=.77)
        save(fig, output, "byzantine_morph_selection_amplification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "docs/figure_data/byzantine_validation.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    plot(json.loads(args.snapshot.read_text(encoding="utf-8")), args.output_dir)
    print(f"Generated two Byzantine PNG/SVG pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
