"""Inspection tests use synthetic labels and never download a dataset."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdl.data import inspect_partition
from tdl.data.inspect_partition import summarize_partition


def test_summary_counts_proportions_and_distance() -> None:
    summary = summarize_partition([2, 2, 9, 9], {0: [0, 1], 1: [2, 3]})
    assert summary["coverage_complete"]
    assert summary["total_assigned_samples"] == 4
    assert summary["num_clients"] == 2
    assert summary["clients"][0]["class_counts"] == {"2": 2, "9": 0}
    assert summary["clients"][1]["class_proportions"] == {"2": 0.0, "9": 1.0}
    assert summary["mean_total_variation_from_global"] == 0.5


@pytest.mark.parametrize("partition", [{0: [0, 0]}, {0: [0]}])
def test_summary_detects_incomplete_or_duplicate_coverage(partition: dict) -> None:
    assert not summarize_partition([0, 1], partition)["coverage_complete"]


def test_summary_handles_empty_client() -> None:
    summary = summarize_partition([0], {0: [0], 1: []})
    assert summary["coverage_complete"]
    assert summary["clients"][1]["total_variation_from_global"] is None
    assert summary["mean_total_variation_from_global"] == 0.0


@pytest.mark.parametrize("index", [-1, 2, 0.5, True])
def test_summary_rejects_invalid_indices(index: int) -> None:
    with pytest.raises(ValueError, match="indices"):
        summarize_partition([0, 1], {0: [index]})


def test_inspection_loads_train_only_and_writes_repeatable_json(tmp_path, monkeypatch, capsys) -> None:
    calls = []

    def fake_dataset(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(targets=torch.tensor(np.repeat([0, 1], 40)))

    monkeypatch.setattr(inspect_partition.datasets, "FashionMNIST", fake_dataset)
    config = tmp_path / "partition.yaml"
    config.write_text("seed: 42\nnum_clients: 4\nalpha: 0.3\nmin_samples_per_client: 1\ndata_dir: data\n")
    output = tmp_path / "results" / "summary.json"
    first = inspect_partition.run(config, output, alpha=10.0)
    first_bytes = output.read_bytes()
    assert json.loads(first_bytes) == first
    assert inspect_partition.run(config, output, alpha=10.0) == first
    assert output.read_bytes() == first_bytes
    assert first["alpha"] == 10.0
    assert first["total_assigned_samples"] == 80
    assert first["minimum_samples_met"] and first["reproducibility_verified"]
    assert all(call == {"root": "data", "train": True, "download": True} for call in calls)
    assert "complete coverage: True" in capsys.readouterr().out
