"""Cheap tests for centralized training utilities."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tdl.training.centralized import build_optimizer, resolve_device, train_one_epoch
from tdl.training.evaluate import evaluate


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_build_optimizer_rejects_unknown_name() -> None:
    model = nn.Linear(2, 2)
    config = {"optimizer": "Adam", "learning_rate": 0.01, "momentum": 0.9}

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(model, config)


def test_train_and_evaluate_with_synthetic_data() -> None:
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    data_loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = nn.Linear(2, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    training_loss = train_one_epoch(
        model, data_loader, criterion, optimizer, device
    )
    test_loss, accuracy = evaluate(model, data_loader, criterion, device)

    assert training_loss > 0.0
    assert test_loss > 0.0
    assert 0.0 <= accuracy <= 1.0
