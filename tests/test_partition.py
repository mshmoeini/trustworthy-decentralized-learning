"""Tests of partition invariants and deterministic rejection sampling."""

import numpy as np
import pytest

from tdl.data.partition import create_dirichlet_partition


LABELS = np.repeat(np.arange(10), 100)


def test_reproducible_and_independent_of_global_rng() -> None:
    first = create_dirichlet_partition(LABELS, 10, 0.3, 42)
    np.random.seed(7)
    assert first == create_dirichlet_partition(LABELS, 10, 0.3, 42)
    assert first != create_dirichlet_partition(LABELS, 10, 0.3, 43)


@pytest.mark.parametrize("alpha", [10.0, 1.0, 0.3])
def test_coverage_no_overlap_client_count_and_minimum(alpha: float) -> None:
    partition = create_dirichlet_partition(LABELS, 10, alpha, 42, 5)
    indices = [index for client in partition.values() for index in client]
    assert set(partition) == set(range(10))
    assert sorted(indices) == list(range(len(LABELS)))
    assert len(set(indices)) == len(indices)
    assert all(len(client) >= 5 for client in partition.values())


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf"), True])
def test_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        create_dirichlet_partition(LABELS, 10, alpha, 42)


@pytest.mark.parametrize("clients", [0, -1, 1.5, True])
def test_invalid_client_count(clients: int) -> None:
    with pytest.raises(ValueError, match="num_clients"):
        create_dirichlet_partition(LABELS, clients, 1.0, 42)


@pytest.mark.parametrize("labels", [[], [[0, 1]], [0.5, 1.0]])
def test_invalid_labels(labels: list) -> None:
    with pytest.raises(ValueError, match="labels"):
        create_dirichlet_partition(labels, 2, 1.0, 42)


@pytest.mark.parametrize("kwargs", [
    {"min_samples_per_client": -1}, {"min_samples_per_client": 1.5},
    {"seed": -1}, {"max_attempts": 0},
])
def test_invalid_controls(kwargs: dict) -> None:
    options = {"seed": 42, **kwargs}
    with pytest.raises(ValueError):
        create_dirichlet_partition(LABELS, 10, 1.0, **options)


def test_impossible_minimum() -> None:
    with pytest.raises(ValueError, match="Not enough samples"):
        create_dirichlet_partition([0, 1], 3, 1.0, 42)


def test_zero_minimum_allows_empty_clients() -> None:
    partition = create_dirichlet_partition([3], 4, 0.3, 42, 0)
    assert sorted(index for indices in partition.values() for index in indices) == [0]
    assert len(partition) == 4


def test_single_client_and_nonconsecutive_classes() -> None:
    partition = create_dirichlet_partition([2, 9, 2, 9], 1, 0.3, 42)
    assert sorted(partition[0]) == [0, 1, 2, 3]


def test_bounded_retries_and_deterministic_success() -> None:
    labels = [0] * 8
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        create_dirichlet_partition(labels, 4, 1.0, 42, 2, max_attempts=1)
    first = create_dirichlet_partition(labels, 4, 1.0, 42, 2)
    assert all(len(indices) == 2 for indices in first.values())
    assert first == create_dirichlet_partition(labels, 4, 1.0, 42, 2)
