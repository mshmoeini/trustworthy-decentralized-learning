"""Neighbor scope, synchronous snapshots, consensus, and isolated training."""

from copy import deepcopy
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tdl.decentralized.aggregation import aggregate_neighborhoods
from tdl.decentralized.consensus import mean_pairwise_rms_parameter_distance
from tdl.decentralized.topology import build_topology
from tdl.decentralized import training
from tdl.federated.training import run_fedavg_round
from tdl.seed import set_seed


CONFIG = {"seed": 42, "local_epochs": 1, "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.9}
DEVICE = torch.device("cpu")


def loaders():
    result = {}
    for node in range(5):
        labels = torch.arange(node + 2) % 2
        inputs = torch.nn.functional.one_hot(labels, 2).float()
        result[node] = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=True,
                                   generator=torch.Generator().manual_seed(node))
    return result


def test_neighbors_only_weighted_and_immutable():
    graph = build_topology("ring_degree_2", 5, 42)
    states = {i: {"weight": torch.tensor([float(i)])} for i in range(5)}
    counts = {i: i + 1 for i in range(5)}
    before = deepcopy(states)
    result = aggregate_neighborhoods(states, counts, graph)
    assert result[0]["weight"].item() == pytest.approx((0 * 1 + 1 * 2 + 4 * 5) / 8)
    states[2]["weight"].fill_(100000)
    assert torch.equal(result[0]["weight"], aggregate_neighborhoods(states, counts, graph)[0]["weight"])
    result[0]["weight"].zero_()
    assert all(torch.equal(states[i]["weight"], before[i]["weight"]) for i in [0, 1, 3, 4])


def test_frozen_snapshot_order_independent():
    graph = build_topology("ring_degree_2", 5, 42)
    states = {i: {"w": torch.tensor([float(i)])} for i in range(5)}
    counts = {i: 1 for i in range(5)}
    before = deepcopy(states)
    forward = aggregate_neighborhoods(states, counts, graph)
    reverse = aggregate_neighborhoods(states, counts, graph, [4, 3, 2, 1, 0])
    assert all(torch.equal(forward[i]["w"], reverse[i]["w"]) for i in states)
    assert all(torch.equal(states[i]["w"], before[i]["w"]) for i in states)
    assert forward[1]["w"].item() == pytest.approx(1.0)


def test_known_consensus_formula_and_pair_order():
    models = {i: nn.Linear(1, 1, bias=False) for i in range(3)}
    with torch.no_grad():
        for model, value in zip(models.values(), [1, 3, 5]):
            model.weight.fill_(value)
    assert mean_pairwise_rms_parameter_distance(models) == pytest.approx(8 / 3)
    assert mean_pairwise_rms_parameter_distance({0: models[2], 1: models[0], 2: models[1]}) == pytest.approx(8 / 3)


def test_consensus_counts_elements_not_layers_and_ignores_buffers():
    a, b = nn.Linear(2, 1), nn.Linear(2, 1)
    with torch.no_grad():
        a.weight.zero_(); a.bias.zero_()
        b.weight.copy_(torch.tensor([[2., 4.]])); b.bias.fill_(6)
    a.register_buffer("floating_buffer", torch.tensor([1000.]))
    b.register_buffer("floating_buffer", torch.tensor([0.]))
    assert mean_pairwise_rms_parameter_distance({0: a, 1: b}) == pytest.approx(math.sqrt(56 / 3))
    b.load_state_dict(a.state_dict())
    b.floating_buffer.zero_()
    assert mean_pairwise_rms_parameter_distance({0: a, 1: b}) == 0


def test_initialization_is_identical_and_isolated():
    initial = nn.Linear(2, 2)
    models = training.initialize_node_models(initial, 5)
    assert mean_pairwise_rms_parameter_distance(models) == 0
    assert len({m.weight.data_ptr() for m in models.values()}) == 5
    before = deepcopy(models[1].state_dict())
    with torch.no_grad():
        models[0].weight.add_(10)
    assert all(torch.equal(before[key], models[1].state_dict()[key]) for key in before)
    assert mean_pairwise_rms_parameter_distance(models) > 0


def test_round_freezes_all_local_updates_and_changes_parameters(monkeypatch):
    set_seed(42)
    models = training.initialize_node_models(nn.Linear(2, 2), 5)
    initial = {node: deepcopy(model.state_dict()) for node, model in models.items()}
    states = {}
    original = training.train_client

    def observe(model, loader, config, device):
        node = len(states)
        assert all(torch.equal(initial[i][key], live.state_dict()[key])
                   for i, live in models.items() for key in initial[i])
        result = original(model, loader, config, device)
        assert any(not torch.equal(initial[node][key], model.state_dict()[key]) for key in initial[node])
        states[node] = deepcopy(model.state_dict())
        return result

    monkeypatch.setattr(training, "train_client", observe)
    graph = build_topology("ring_degree_2", 5, 42)
    metrics = training.run_decentralized_round(models, loaders(), graph, CONFIG, DEVICE, 1)
    expected = aggregate_neighborhoods(states, {i: i + 2 for i in models}, graph)
    assert metrics["num_nodes"] == 5 and metrics["total_training_samples"] == 20
    assert all(torch.equal(models[i].state_dict()[key], expected[i][key]) for i in models for key in expected[i])


@pytest.mark.parametrize("stochastic", [False, True])
def test_training_and_aggregation_execution_order_do_not_change_results(stochastic):
    def simulate(order):
        set_seed(42)
        initial = (nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.4), nn.Linear(4, 2))
                   if stochastic else nn.Linear(2, 2))
        models = training.initialize_node_models(initial, 5)
        clients = loaders()
        graph = build_topology("ring_degree_2", 5, 42)
        metrics = [training.run_decentralized_round(models, clients, graph, CONFIG, DEVICE, r, order)
                   for r in [1, 2]]
        return models, metrics
    forward, f_metrics = simulate([0, 1, 2, 3, 4])
    reverse, r_metrics = simulate([4, 3, 2, 1, 0])
    assert f_metrics == r_metrics
    assert all(torch.equal(forward[i].state_dict()[key], reverse[i].state_dict()[key])
               for i in forward for key in forward[i].state_dict())


def test_fully_connected_matches_fedavg_states_every_round():
    set_seed(42)
    global_model = nn.Linear(2, 2)
    models = training.initialize_node_models(global_model, 5)
    fed_loaders, dec_loaders = loaders(), loaders()
    graph = build_topology("fully_connected", 5, 42)
    for round_number in [1, 2]:
        fed = run_fedavg_round(global_model, fed_loaders, CONFIG, DEVICE, round_number)
        dec = training.run_decentralized_round(models, dec_loaders, graph, CONFIG, DEVICE, round_number)
        assert fed["weighted_mean_client_training_loss"] == dec["weighted_mean_local_training_loss"]
        assert mean_pairwise_rms_parameter_distance(models) == 0
        assert all(torch.equal(model.state_dict()[key], global_model.state_dict()[key])
                   for model in models.values() for key in global_model.state_dict())
