"""Outgoing attack correctness, communication timing, isolation and controls."""
from copy import deepcopy
from functools import partial

import networkx as nx
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from tdl.byzantine import attacks, diagnostics, runner as attack_runner
from tdl.decentralized import epidemic, morph, runner, training


@pytest.mark.parametrize("name,strength,expected", [
    ("none", 1., [2., 3.]), ("sign_flip", 1., [0., -1.]),
    ("sign_flip", 2., [-1., -3.]), ("scaling", 5., [6., 11.])])
def test_update_definition_and_owned_payload(name, strength, expected):
    start = {"w": torch.tensor([1., 1.]), "counter": torch.tensor(2)}
    local = {"w": torch.tensor([2., 3.]), "counter": torch.tensor(3)}
    original_start, original_local = deepcopy(start), deepcopy(local)
    result = attacks.apply_attack(name, start, local, strength)
    torch.testing.assert_close(result["w"], torch.tensor(expected), rtol=0, atol=0)
    assert result["counter"].item() == 3
    result["w"].zero_(); result["counter"].zero_()
    for original, actual in [(original_start, start), (original_local, local)]:
        assert all(torch.equal(actual[k], v) for k, v in original.items())


def test_only_configured_senders_transform_and_inputs_stay_frozen():
    start = {n: {"w": torch.ones(2)} for n in range(3)}
    local = {n: {"w": torch.tensor([2., 3.])} for n in range(3)}
    result = attacks.outgoing_payloads(start, local, {"byzantine_nodes": [0], "attack": "sign_flip"})
    assert torch.equal(result[0]["w"], torch.tensor([0., -1.]))
    assert all(torch.equal(result[n]["w"], local[n]["w"]) for n in [1, 2])
    result[1]["w"].zero_()
    assert all(torch.equal(s["w"], torch.tensor([2., 3.])) for s in local.values())


@pytest.mark.parametrize("config", [
    {"attack": "unknown"}, {"attack_strength": -1}, {"attack_strength": float('nan')},
    {"attack_strength": True}, {"byzantine_nodes": [True]}, {"byzantine_nodes": [0, 0]},
    {"byzantine_nodes": [9]}, {"byzantine_nodes": "0"}])
def test_invalid_attack_configuration(config):
    with pytest.raises(ValueError):
        attacks.validate_attack_config(config, range(4))


@pytest.mark.parametrize("local", [{}, {"w": torch.ones(3)}, {"w": torch.ones(2, dtype=torch.float64)},
                                    {"w": torch.tensor([float('nan'), 1.])}])
def test_invalid_attack_states(local):
    with pytest.raises(ValueError):
        attacks.apply_attack("sign_flip", {"w": torch.ones(2)}, local)


def test_attack_overflow_is_reported_without_clipping():
    with pytest.raises(ValueError, match="overflow"):
        attacks.apply_attack("scaling", {"w": torch.zeros(1)}, {"w": torch.tensor([1e38])}, 100.)


def config(**overrides):
    c = dict(seed=42, num_clients=4, alpha=1., data_dir="unused", device="cpu", rounds=2,
        local_epochs=1, batch_size=4, num_workers=0, min_samples_per_client=1,
        optimizer="SGD", learning_rate=.1, momentum=.9, byzantine_nodes=[0],
        attack="sign_flip", attack_strength=1., k=2, beta=10., random_peer_count=1,
        topology_refresh_interval=1, selection_mode="paper_resample", outgoing_capacity=None,
        morph_mode="morph_code_faithful", bootstrap_topology="ring_degree_2",
        epidemic_k=2, aggregation_mode="paper_epidemic")
    c.update(overrides)
    return c


def nodes():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    models = training.initialize_node_models(model, 4)
    dataset = TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long))
    loaders = {n: DataLoader(dataset, batch_size=2, shuffle=True,
        generator=torch.Generator().manual_seed(n)) for n in models}
    return models, loaders


def one_round(algorithm, models, loaders, c, number=1, protocol=None, order=None):
    if algorithm == "el":
        return epidemic.run_epidemic_round(models, loaders, c, torch.device("cpu"), number, order)[0]
    return morph.run_morph_round(models, loaders, protocol, c, torch.device("cpu"), number, order,
        selection_observer=partial(diagnostics.observe_candidates, byzantine_nodes=c.get("byzantine_nodes", [])))


@pytest.mark.parametrize("algorithm", ["el", "morph"])
def test_attack_after_all_training_before_exchange_with_honest_self_and_frozen_install(algorithm, monkeypatch):
    models, loaders = nodes()
    original = {n: deepcopy(m.state_dict()) for n, m in models.items()}
    trained, seen = [], {}
    def normal_training(local, loader, c, device):
        assert all(torch.equal(models[n].weight, original[n]["weight"]) for n in models)
        with torch.no_grad():
            local.weight.add_(torch.tensor([[1., 2.]]))
        trained.append(1)
        return {"num_samples": len(loader.dataset), "training_loss": 1., "epoch_losses": [1.]}
    monkeypatch.setattr(training, "train_client", normal_training)
    module = epidemic if algorithm == "el" else morph
    boundary = module.outgoing_payloads
    def observed_boundary(starts, snapshots, c):
        assert len(trained) == 4
        assert all(torch.equal(s["weight"], torch.tensor([[2., 3.]])) for s in snapshots.values())
        seen["honest"] = deepcopy(snapshots)
        result = boundary(starts, snapshots, c)
        assert torch.equal(result[0]["weight"], torch.tensor([[0., -1.]]))
        assert all(torch.equal(result[n]["weight"], snapshots[n]["weight"]) for n in [1, 2, 3])
        return result
    monkeypatch.setattr(module, "outgoing_payloads", observed_boundary)
    aggregate = module.aggregate_incoming
    def observed_aggregate(payload, counts, graph, *args, **kwargs):
        assert len(trained) == 4 and "honest" in seen
        assert all(torch.equal(models[n].weight, original[n]["weight"]) for n in models)
        own = kwargs["self_snapshot"]
        assert all(torch.equal(own[n]["weight"], seen["honest"][n]["weight"]) for n in models)
        if algorithm == "morph":
            for receiver in graph.successors(0):
                assert torch.equal(protocol.views[receiver].models[0]["weight"], payload[0]["weight"])
        updates = aggregate(payload, counts, graph, *args, **kwargs)
        for receiver in graph:
            expected = (own[receiver]["weight"] + sum((payload[s]["weight"] for s in graph.predecessors(receiver)), torch.zeros(1, 2))) / (1 + graph.in_degree(receiver))
            torch.testing.assert_close(updates[receiver]["weight"], expected)
        assert all(torch.equal(own[n]["weight"], seen["honest"][n]["weight"]) for n in models)
        seen["updates"] = updates
        return updates
    monkeypatch.setattr(module, "aggregate_incoming", observed_aggregate)
    c = config()
    protocol = morph.MorphProtocol(nx.cycle_graph(4), c) if algorithm == "morph" else None
    metrics = one_round(algorithm, models, loaders, c, protocol=protocol)
    assert metrics["byzantine_deliveries"]["poisoned_model_deliveries"] == 2
    assert all(torch.equal(m.weight, seen["updates"][n]["weight"]) for n, m in models.items())


@pytest.mark.parametrize("algorithm", ["el", "morph"])
def test_disabled_attack_preserves_clean_training_graphs_and_weights(algorithm, monkeypatch):
    def train(local, loader, c, device):
        with torch.no_grad():
            local.weight.add_(torch.rand_like(local.weight))
        return {"num_samples": 4, "training_loss": 1., "epoch_losses": [1.]}
    monkeypatch.setattr(training, "train_client", train)
    clean, clean_loaders = nodes(); observed, observed_loaders = nodes()
    c = config(); c.pop("attack"); c.pop("byzantine_nodes")
    no_attack = dict(c, attack="none", byzantine_nodes=[0])
    p = morph.MorphProtocol(nx.cycle_graph(4), c)
    q = morph.MorphProtocol(nx.cycle_graph(4), no_attack)
    for number in [1, 2, 3]:
        a = one_round(algorithm, clean, clean_loaders, c, number, p)
        b = one_round(algorithm, observed, observed_loaders, no_attack, number, q, [3, 2, 1, 0])
        assert a["nodes"] == b["nodes"]
        assert all(torch.equal(clean[n].weight, observed[n].weight) for n in clean)
        if algorithm == "morph":
            assert a["topology"] == b["topology"]
        assert b["byzantine_deliveries"]["poisoned_model_deliveries"] == 0


def test_candidate_observer_ranks_local_scores_without_rescoring_or_mutating():
    view = morph.PeerView(1, [0, 2, 3])
    view.seen_real.add(0); view.score_sources[0] = "direct"; view.score_cached[0] = True
    scores = {1: {0: -.5, 2: .8, 3: -.5}}
    original = deepcopy(scores)
    result = diagnostics.observe_candidates({1: view}, scores, {}, set(), {(0, 1)}, [0])
    record = result["receivers"][0]
    assert record["rank_min"] == 1 and record["rank_max"] == 2
    assert record["relative_rank"] == 0 and record["guided_proposal"]
    assert result["honest_nodes_with_direct_receipt_of_any_attacker"] == 1
    assert scores == original and not view.cache


@pytest.mark.parametrize("algorithm", ["el", "morph"])
def test_attack_runner_end_to_end_matched_control_and_determinism(tmp_path, monkeypatch, algorithm):
    labels = torch.tensor([0, 1] * 20)
    dataset = TensorDataset(torch.nn.functional.one_hot(labels, 2).float(), labels)
    dataset.targets = labels
    monkeypatch.setattr(runner, "load_fashion_mnist", lambda _: (dataset, dataset))
    monkeypatch.setattr(runner, "FashionMNISTCNN", lambda: nn.Linear(2, 2))
    c = config(attack="none")
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(c))
    clean = tmp_path / "clean.json"
    attack_runner.run(path, clean, algorithm)
    a = attack_runner.run(path, tmp_path / "attack.json", algorithm, clean, "sign_flip", 1.)
    b = attack_runner.run(path, tmp_path / "repeat.json", algorithm, clean, "sign_flip", 1.)
    assert a == b and a["status"] == "complete"
    assert a["metadata"]["byzantine_nodes"] == [0]
    assert len(a["rounds"]) == 2
    assert all(r["directed_model_transmissions"] == 8 for r in a["rounds"])
    assert all(r["byzantine_impact"]["honest_only_mean_accuracy"] is not None for r in a["rounds"])
    if algorithm == "morph":
        assert all(len(r["peer_selection_observations"]["receivers"]) == 3 for r in a["rounds"])
    with pytest.raises(FileExistsError):
        attack_runner.run(path, tmp_path / "attack.json", algorithm)
    c["learning_rate"] = .2; path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match="configuration"):
        attack_runner.run(path, tmp_path / "mismatch.json", algorithm, clean)
