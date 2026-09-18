"""Attack-only Morph/EL development runner with immutable clean controls."""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

import torch

from tdl.config import load_config
from tdl.byzantine.attacks import ATTACKS, validate_attack_config
from tdl.byzantine.diagnostics import observe_candidates, delivery_summary, impact_summary
from tdl.decentralized import epidemic, morph
from tdl.decentralized.runner import prepare_experiment, evaluate_nodes
from tdl.decentralized.topology import build_topology

ROOT = Path(__file__).resolve().parents[3]
METRICS = ("mean_node_test_accuracy", "worst_node_test_accuracy", "node_test_accuracy_std",
           "mean_node_test_loss", "mean_pairwise_rms_parameter_distance")
ATTACK_KEYS = {"attack", "attack_strength", "byzantine_nodes"}


def clean_selection_frequency(baseline, algorithm, ids):
    """Replay discovery/seeded random reservations from recorded clean graphs.

    No model/score reconstruction or training. Candidate identities are recovered
    by the original frozen peer-list gossip, and verified against saved counts.
    This identifies clean guided versus random accepted edges exactly for the
    primary paper_resample profile with refresh every round.
    """
    config = baseline["config"]
    result = []
    if algorithm == "morph":
        if config["selection_mode"] != "paper_resample" or config["topology_refresh_interval"] != 1:
            raise ValueError("Clean guided-edge reconstruction requires the validated refresh profile")
        bootstrap = build_topology(config["bootstrap_topology"], config["num_clients"], config["seed"], config.get("small_world_rewire_probability", .2))
        known = {n: set(bootstrap.neighbors(n)) for n in bootstrap}
    for round_metrics in baseline["rounds"]:
        graph = torch_graph(round_metrics["topology"]["edge_list"], config["num_clients"])
        record = delivery_summary(graph, ids, False)
        if algorithm == "morph":
            reserved = set()
            for receiver in sorted(known):
                rng = random.Random((config["seed"] + round_metrics["round"]) * len(known) + receiver)
                reserved.update((s, receiver) for s in rng.sample(sorted(known[receiver]), config["random_peer_count"]))
            edges = set(graph.edges())
            if not reserved <= edges or any(s not in known[r] for s, r in edges):
                raise ValueError("Recorded clean graph disagrees with local-view random reservations")
            record["guided_selections_from_attackers_to_honest"] = sum(s in ids and r not in ids for s, r in edges - reserved)
            record["random_selections_from_attackers_to_honest"] = sum(s in ids and r not in ids for s, r in edges & reserved)
            for sender, receiver in edges:
                known[sender].add(receiver)
            metadata = {n: set(peers) for n, peers in known.items()}
            for sender, receiver in edges:
                known[receiver].update((metadata[sender] | {sender}) - {receiver})
            if {str(n): len(peers) for n, peers in known.items()} != round_metrics["known_peer_counts"]:
                raise ValueError("Clean discovery replay differs from recorded counts")
        result.append({"round": round_metrics["round"], **record})
    return result


def torch_graph(edges, num_nodes):
    # Graph construction is report bookkeeping, never candidate selection.
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    return graph


def validate_baseline(config, algorithm, baseline_path):
    raw = Path(baseline_path).read_bytes()
    baseline = json.loads(raw)
    clean_config = {k: v for k, v in config.items() if k not in ATTACK_KEYS}
    baseline_config = {k: v for k, v in baseline["config"].items() if k not in ATTACK_KEYS}
    if baseline["config"].get("attack", "none") != "none" or baseline.get("status", "complete") != "complete":
        raise ValueError("Control must be a completed no-attack baseline")
    if baseline_config != clean_config or len(baseline["rounds"]) != config["rounds"]:
        raise ValueError("Clean baseline must match every non-attack configuration field")
    if algorithm == "morph" and baseline["metadata"]["morph_mode"] != "morph_code_faithful":
        raise ValueError("Expected primary Morph clean control")
    if algorithm == "el" and baseline["metadata"]["aggregation_rule"] != "uniform_self_plus_incoming":
        raise ValueError("Expected uniform EL clean control")
    hashes = baseline["metadata"]["local_implementation"]["source_sha256"]
    modified = {"src/tdl/decentralized/morph.py", "src/tdl/decentralized/epidemic.py"}
    for name, expected in hashes.items():
        if name not in modified and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Controlling clean baseline source changed: {name}")
    return baseline, {"file": Path(baseline_path).name, "sha256": hashlib.sha256(raw).hexdigest(),
        "metadata": baseline["metadata"], "config": baseline["config"],
        "reuse_note": "Exact clean config/partition/initial evaluation checks. Shared training/model/data/optimizer sources unchanged; Morph/EL changes are attack boundary and observational hooks, with no-attack equivalence tests. EL42 historical Morph hash is its observational cosine helper."}


def run(config_path, output, algorithm, baseline_path=None, attack=None, strength=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserve existing report: {output}")
    if algorithm not in ("morph", "el"):
        raise ValueError("Use morph or el")
    config = load_config(config_path)
    if attack is not None:
        config["attack"] = attack
    if strength is not None:
        config["attack_strength"] = strength
    name, strength, ids = validate_attack_config(config, range(config["num_clients"]))
    if len(ids) >= config["num_clients"]:
        raise ValueError("Development impact reporting requires honest nodes")
    if algorithm == "morph":
        morph.validate_config(config)
        if morph.protocol_mode(config) != "morph_code_faithful":
            raise ValueError("Attack development uses only primary Morph")
    else:
        epidemic.validate_protocol(config["num_clients"], config["epidemic_k"], config["seed"], 1)
        if config["aggregation_mode"] != "paper_epidemic":
            raise ValueError("Attack development uses uniform EL")
    for key, minimum in [("rounds", 1), ("local_epochs", 1), ("batch_size", 1), ("num_workers", 0), ("min_samples_per_client", 1)]:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {key}")
    source_names = sorted({p.relative_to(ROOT).as_posix() for p in (ROOT / "src/tdl").rglob("*.py")})
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in source_names}
    baseline, provenance = validate_baseline(config, algorithm, baseline_path) if baseline_path else (None, None)
    models, loaders, test_loader, device, partition, train_size, test_size = prepare_experiment(config)
    initial = evaluate_nodes(models, test_loader, device)
    if baseline:
        if partition != baseline["partition"] or any(initial[k] != baseline["initial_node_evaluation"][k] for k in METRICS):
            raise ValueError("Clean baseline partition or initialization differs")
    protocol = None
    observer = None
    if algorithm == "morph":
        bootstrap = build_topology(config["bootstrap_topology"], config["num_clients"], config["seed"], config.get("small_world_rewire_probability", .2))
        protocol = morph.MorphProtocol(bootstrap, config)
        observer = partial(observe_candidates, byzantine_nodes=ids)
    report = {"status": "running", "metadata": {"algorithm": "morph_code_faithful" if algorithm == "morph" else "EL-Local uniform",
        "attack": name, "attack_strength": strength, "byzantine_nodes": list(ids),
        "attack_definition": "delta=theta_local-theta_start; sent model=theta_start-lambda*delta (sign_flip), theta_start+lambda*delta (scaling)",
        "attack_timing": "Capture start before normal training; transform frozen post-training outgoing states; peer selection uses honest local weights and previously received payloads; exchange malicious payloads; self contribution stays honest; compute all aggregates before installation",
        "attack_scope": "Same malicious outgoing payload to every requesting receiver; no equivocation, data poisoning, metadata forgery or defenses",
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "training_samples": train_size, "test_samples": test_size,
        "morph_mode": "morph_code_faithful" if algorithm == "morph" else None,
        "aggregation_rule": "uniform" if algorithm == "morph" else "uniform_self_plus_incoming", "traffic_definition": "Actual directed full-model deliveries; excludes metadata/negotiation",
        "development_only": True, "interpretation": "Single seed, one attacker development check; no general robustness/vulnerability or causal selection claim",
        "local_implementation": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "state": "Local uncommitted attack development; git_head is checkpoint base", "source_sha256": hashes,
            "config_file_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "effective_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}},
        "config": config, "partition": partition, "initial_node_evaluation": initial,
        "clean_baseline": provenance, "clean_peer_selection_frequency": clean_selection_frequency(baseline, algorithm, ids) if baseline else None,
        "rounds": []}
    previous = None
    print(f"START {algorithm} {name} lambda={strength} ids={list(ids)} device={device}", flush=True)
    try:
        for number in range(1, config["rounds"] + 1):
            if algorithm == "morph":
                metrics = morph.run_morph_round(models, loaders, protocol, config, device, number, selection_observer=observer)
                graph = protocol.graph
                metrics["peer_selection_observations"].update(
                    guided_accepted_attacker_to_honest=sum(s in ids and r not in ids for s, r in set(graph.edges()) - protocol.random_edges),
                    random_accepted_attacker_to_honest=sum(s in ids and r not in ids for s, r in protocol.random_edges))
            else:
                metrics, graph = epidemic.run_epidemic_round(models, loaders, config, device, number)
                metrics["topology"] = epidemic.communication_summary(graph, previous)
                metrics["directed_model_transmissions"] = graph.number_of_edges()
            evaluation = evaluate_nodes(models, test_loader, device)
            if any(not math.isfinite(evaluation[k]) for k in METRICS):
                raise RuntimeError("Nonfinite evaluation metric")
            impact = impact_summary(evaluation, ids)
            by_id = {n["node_id"]: n for n in evaluation.pop("nodes")}
            for node in metrics["nodes"]:
                node.update(by_id[node["node_id"]])
            metrics.update(evaluation)
            metrics["byzantine_impact"] = impact
            if baseline:
                clean = baseline["rounds"][number - 1]
                metrics["attack_minus_clean"] = {k: metrics[k] - clean[k] for k in METRICS}
                clean_impact = impact_summary(clean, ids)
                metrics["honest_attack_minus_clean"] = {k: impact[k] - clean_impact[k] for k in
                    ["honest_only_mean_accuracy", "honest_only_worst_accuracy", "honest_only_accuracy_std"]}
                metrics["clean_honest_impact"] = clean_impact
            report["rounds"].append(metrics)
            previous = graph.copy()
            print(f"Round {number} | mean={metrics['mean_node_test_accuracy']:.3%} | honest={impact['honest_only_mean_accuracy']:.3%} | poisoned={metrics['byzantine_deliveries']['poisoned_model_deliveries']} | tx={metrics['directed_model_transmissions']}", flush=True)
        if any(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != h for n, h in hashes.items()):
            raise RuntimeError("Implementation changed during experiment")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", failed_round=len(report["rounds"]) + 1,
                      failure={"type": type(error).__name__, "message": str(error)})
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=["morph", "el"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--attack", choices=ATTACKS)
    parser.add_argument("--strength", type=float)
    args = parser.parse_args()
    run(args.config, args.output, args.algorithm, args.baseline, args.attack, args.strength)


if __name__ == "__main__":
    main()
