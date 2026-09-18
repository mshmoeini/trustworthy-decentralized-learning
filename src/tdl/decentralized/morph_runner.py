"""Small Morph development runner; refuses to overwrite local reports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch

from tdl.config import load_config
from tdl.decentralized.morph import MorphProtocol, SOURCE_REVISION, run_morph_round, validate_config, protocol_mode
from tdl.decentralized.runner import prepare_experiment, evaluate_nodes
from tdl.decentralized.topology import build_topology


def run(config_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserve existing report: {output}")
    config = load_config(config_path)
    validate_config(config)
    root = Path(__file__).resolve().parents[3]
    source_names = ["src/tdl/decentralized/morph.py", "src/tdl/decentralized/morph_runner.py",
                    "src/tdl/decentralized/training.py", "src/tdl/decentralized/epidemic.py",
                    "src/tdl/decentralized/runner.py", "src/tdl/decentralized/topology.py",
                    "src/tdl/federated/training.py", "src/tdl/federated/aggregation.py",
                    "src/tdl/models/cnn.py", "src/tdl/data/partition.py", "src/tdl/seed.py"]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names}
    bootstrap = build_topology(config["bootstrap_topology"], config["num_clients"], config["seed"], config.get("small_world_rewire_probability", .2))
    protocol = MorphProtocol(bootstrap, config)
    models, loaders, test_loader, device, partition, train_size, test_size = prepare_experiment(config)
    report = {"metadata": {"algorithm": protocol_mode(config), "morph_mode": protocol_mode(config), "aggregation_rule": "uniform", "dataset": "Fashion-MNIST",
        "device": str(device), "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_samples": train_size, "test_samples": test_size, "source_revision": SOURCE_REVISION,
        "source_paper": "https://arxiv.org/abs/2602.03383", "source_repository": "https://github.com/bacox/Morph",
        "semantics_document": "docs/morph.md", "selection_mode": config["selection_mode"],
        "candidate_pool": "local discovered peers only", "similarity": "equally weighted named parameter tensor cosines; skip zero norms",
        "peer_seed_rule": "(seed + one_based_round) * num_nodes + receiver_id; sorted candidates",
        "node_seed_rule": "seed + one_based_round * num_nodes + node_id",
        "snapshot_semantics": "all training, metadata, aggregation barriers; synchronous",
        "traffic_definition": "directed full-model edge transmissions; excludes metadata and negotiation",
        "development_only": True, "interpretation": "Development validation; no statistical claims",
        "source_reconciliation": "morph_code_faithful serves all wanted-sender requests; morph_paper_capped preserves experimental capacity negotiation. Hybrid paper_resample selection is a documented adaptation, not the released code's one-swap path; see docs/morph.md"},
        "config": config, "partition": partition, "initial_node_evaluation": evaluate_nodes(models, test_loader, device),
        "initial_known_peer_counts": {str(n): len(v.known) for n, v in protocol.views.items()}, "rounds": []}
    report["metadata"]["local_implementation"] = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
        "state": "uncommitted Morph implementation; git_head is the base, not an implementation commit",
        "source_sha256": source_hashes,
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest()}
    print(f"Device: {device} | {protocol_mode(config)} {config['selection_mode']} | alpha={config['alpha']}", flush=True)
    for number in range(1, config["rounds"] + 1):
        metrics = run_morph_round(models, loaders, protocol, config, device, number)
        evaluation = evaluate_nodes(models, test_loader, device)
        nodes = {n["node_id"]: n for n in evaluation.pop("nodes")}
        for node in metrics["nodes"]:
            node.update(nodes[node["node_id"]])
        metrics.update(evaluation)
        report["rounds"].append(metrics)
        print(f"Round {number} | mean={metrics['mean_node_test_accuracy']:.2%} | worst={metrics['worst_node_test_accuracy']:.2%} | transmissions={metrics['directed_model_transmissions']} | declines={metrics['declined_requests']}", flush=True)
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != value for name, value in source_hashes.items()):
        raise RuntimeError("Implementation files changed during the run; report provenance is invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/morph_code_faithful.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/morph_code_faithful_alpha_0.3.json"))
    args = parser.parse_args()
    run(args.config, args.output)
