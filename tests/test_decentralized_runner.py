"""Synthetic integration tests of the runner and equal-node evaluation metrics."""

import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from tdl.decentralized import runner


def config_file(tmp_path, rounds=2):
    path = tmp_path / "decentralized.yaml"
    path.write_text(f"seed: 42\nnum_clients: 5\nalpha: 0.3\nmin_samples_per_client: 1\n"
                    f"data_dir: data\nbatch_size: 4\nlocal_epochs: 1\nrounds: {rounds}\n"
                    "optimizer: SGD\nlearning_rate: 0.1\nmomentum: 0.9\ndevice: cpu\nnum_workers: 0\n"
                    "topology: ring_degree_2\nsmall_world_rewire_probability: 0.2\n")
    return path


def test_synthetic_runner_reproducibility_and_node_metrics(tmp_path, monkeypatch):
    labels = torch.tensor([0, 1] * 20)
    dataset = TensorDataset(torch.nn.functional.one_hot(labels, 2).float(), labels)
    dataset.targets = labels
    monkeypatch.setattr(runner, "load_fashion_mnist", lambda _: (dataset, dataset))
    monkeypatch.setattr(runner, "FashionMNISTCNN", lambda: nn.Linear(2, 2))
    config = config_file(tmp_path)
    output = tmp_path / "report.json"
    first = runner.run(config, output, alpha=1.0, topology="fully_connected")
    assert json.loads(output.read_text()) == first
    assert runner.run(config, output, alpha=1.0, topology="fully_connected") == first
    assert first["config"]["alpha"] == 1.0
    assert first["partition"]["coverage_complete"]
    assert first["initial_node_evaluation"]["mean_pairwise_rms_parameter_distance"] == 0
    for metrics in first["rounds"]:
        assert metrics["num_nodes"] == 5 and metrics["total_training_samples"] == 40
        assert metrics["node_test_accuracy_std"] == 0
        assert metrics["mean_pairwise_rms_parameter_distance"] == 0
        assert len(metrics["nodes"]) == 5
        assert all(0 <= node["test_accuracy"] <= 1 and node["test_loss"] > 0
                   and node["local_training_loss"] > 0 for node in metrics["nodes"])
        assert sum(n["num_local_training_samples"] for n in metrics["nodes"]) == 40


@pytest.mark.parametrize("rounds", [0, -1, 1.5])
def test_invalid_rounds_fail_before_loading(tmp_path, monkeypatch, rounds):
    def forbidden(_):
        raise AssertionError("Dataset should not be loaded")
    monkeypatch.setattr(runner, "load_fashion_mnist", forbidden)
    with pytest.raises(ValueError, match="rounds"):
        runner.run(config_file(tmp_path, rounds), tmp_path / "report.json")
