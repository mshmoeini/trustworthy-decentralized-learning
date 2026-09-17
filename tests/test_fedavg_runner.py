"""Full runner tests with small synthetic data, independent of Fashion-MNIST."""

import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from tdl.federated import fedavg


def config_file(tmp_path, rounds=2):
    path = tmp_path / "fedavg.yaml"
    path.write_text(f"seed: 42\nnum_clients: 3\nalpha: 0.3\nmin_samples_per_client: 1\n"
                    f"data_dir: data\nbatch_size: 4\nlocal_epochs: 1\nrounds: {rounds}\n"
                    "optimizer: SGD\nlearning_rate: 0.1\nmomentum: 0.9\ndevice: cpu\nnum_workers: 0\n")
    return path


def test_synthetic_runner_metadata_metrics_and_reproducibility(tmp_path, monkeypatch, capsys):
    labels = torch.tensor([0, 1] * 15)
    inputs = torch.nn.functional.one_hot(labels, num_classes=2).float()
    dataset = TensorDataset(inputs, labels)
    dataset.targets = labels
    monkeypatch.setattr(fedavg, "load_fashion_mnist", lambda _: (dataset, dataset))
    monkeypatch.setattr(fedavg, "FashionMNISTCNN", lambda: nn.Linear(2, 2))
    config = config_file(tmp_path)
    output = tmp_path / "results" / "fedavg.json"
    first = fedavg.run(config, output, alpha=1.0)
    assert json.loads(output.read_text()) == first
    assert fedavg.run(config, output, alpha=1.0) == first
    assert first["metadata"]["device"] == "cpu"
    assert first["metadata"]["config"]["alpha"] == 1.0
    assert first["partition"]["coverage_complete"]
    assert len(first["rounds"]) == 2
    for metrics in first["rounds"]:
        assert metrics["num_clients"] == 3 and metrics["total_samples"] == 30
        assert 0 <= metrics["global_test_accuracy"] <= 1
        assert 0 < metrics["global_test_loss"] < float("inf")
        assert 0 < metrics["weighted_mean_client_training_loss"] < float("inf")
    assert first["rounds"][-1]["global_test_loss"] < first["initial_evaluation"]["global_test_loss"]
    assert "Round 2/2" in capsys.readouterr().out


@pytest.mark.parametrize("rounds", [0, -1, 1.5])
def test_invalid_rounds_fail_before_dataset_loading(tmp_path, monkeypatch, rounds):
    def forbidden(_):
        raise AssertionError("Invalid configuration must fail before loading data")
    monkeypatch.setattr(fedavg, "load_fashion_mnist", forbidden)
    with pytest.raises(ValueError, match="rounds"):
        fedavg.run(config_file(tmp_path, rounds), tmp_path / "report.json")
