"""Synthetic client training, isolation, and full-round tests."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tdl.federated import training
from tdl.federated.aggregation import weighted_average_state_dicts
from tdl.seed import set_seed


CONFIG = {"seed": 42, "local_epochs": 2, "optimizer": "SGD",
          "learning_rate": 0.1, "momentum": 0.9}
DEVICE = torch.device("cpu")


def loaders():
    return {client: DataLoader(TensorDataset(torch.tensor([[1., 0.], [0., 1.], [1., 1.]])[:size],
                                             torch.tensor([0, 1, 0])[:size]),
                              batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(client))
            for client, size in enumerate([2, 3])}


def test_local_training_changes_parameters_and_reports_loss() -> None:
    set_seed(42)
    model = nn.Linear(2, 2)
    before = deepcopy(model.state_dict())
    metrics = training.train_client(model, loaders()[0], CONFIG, DEVICE)
    assert metrics["num_samples"] == 2
    assert len(metrics["epoch_losses"]) == 2
    assert 0 < metrics["training_loss"] < float("inf")
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_clients_share_initial_weights_without_mutating_global(monkeypatch) -> None:
    set_seed(42)
    global_model = nn.Linear(2, 2)
    initial = deepcopy(global_model.state_dict())
    trained = []
    original = training.train_client

    def observe(model, loader, config, device):
        for key, value in initial.items():
            assert torch.equal(model.state_dict()[key], value)
            assert torch.equal(global_model.state_dict()[key], value)
            assert model.state_dict()[key].data_ptr() != global_model.state_dict()[key].data_ptr()
        result = original(model, loader, config, device)
        trained.append(deepcopy(model.state_dict()))
        return result

    monkeypatch.setattr(training, "train_client", observe)
    metrics = training.run_fedavg_round(global_model, loaders(), CONFIG, DEVICE, 1)
    expected = weighted_average_state_dicts(trained, [2, 3])
    assert metrics["num_clients"] == 2 and metrics["total_samples"] == 5
    for key in expected:
        assert torch.equal(global_model.state_dict()[key], expected[key])
    assert metrics["weighted_mean_client_training_loss"] == pytest.approx(
        sum(client["training_loss"] * client["num_samples"] / 5 for client in metrics["clients"]))


def test_two_synthetic_rounds_are_reproducible() -> None:
    def simulation():
        set_seed(42)
        model = nn.Linear(2, 2)
        clients = loaders()
        assert clients[0].generator is not clients[1].generator
        metrics = [training.run_fedavg_round(model, clients, CONFIG, DEVICE, r) for r in [1, 2]]
        return deepcopy(model.state_dict()), metrics

    first, first_metrics = simulation()
    second, second_metrics = simulation()
    assert first_metrics == second_metrics
    assert all(torch.equal(first[key], second[key]) for key in first)


@pytest.mark.parametrize("epochs", [0, -1, 1.5])
def test_invalid_local_epochs(epochs) -> None:
    with pytest.raises(ValueError, match="local_epochs"):
        training.train_client(nn.Linear(2, 2), loaders()[0], {**CONFIG, "local_epochs": epochs}, DEVICE)
