"""Reproducible, class-wise Dirichlet partitions of dataset indices."""

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike


def create_dirichlet_partition(
    labels: ArrayLike,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
    *,
    max_attempts: int = 1000,
) -> dict[int, list[int]]:
    """Assign every sample exactly once using a local seeded NumPy RNG.

    For each class, shuffle its indices, draw symmetric Dirichlet proportions,
    and draw integer client counts from a multinomial with those proportions.
    Smaller alpha favors more concentrated class allocations; larger alpha
    favors similar proportions. Neither guarantees balanced client sizes.

    Reject entire allocations below the minimum and advance the same RNG for
    at most max_attempts. This conditions the distribution on the minimum;
    no samples are moved afterward to force a particular label distribution.
    Labels must be a nonempty, one-dimensional array of integer class IDs.
    """
    for name, value, minimum in (
        ("num_clients", num_clients, 1),
        ("min_samples_per_client", min_samples_per_client, 0),
        ("seed", seed, 0),
        ("max_attempts", max_attempts, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("alpha must be finite and greater than zero.")

    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array.")
    if not np.issubdtype(label_array.dtype, np.integer):
        raise ValueError("labels must contain integer class IDs.")
    if label_array.size < num_clients * min_samples_per_client:
        raise ValueError("Not enough samples to satisfy min_samples_per_client.")

    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(label_array == label) for label in np.unique(label_array)]
    concentration = np.full(num_clients, alpha, dtype=float)
    for _ in range(max_attempts):
        partition: dict[int, list[int]] = {client: [] for client in range(num_clients)}
        for indices in class_indices:
            shuffled = rng.permutation(indices)
            proportions = rng.dirichlet(concentration)
            if not np.all(np.isfinite(proportions)):
                break
            counts = rng.multinomial(len(shuffled), proportions)
            splits = np.split(shuffled, np.cumsum(counts)[:-1])
            for client, split in enumerate(splits):
                partition[client].extend(split.tolist())
        else:
            if all(len(indices) >= min_samples_per_client for indices in partition.values()):
                for indices in partition.values():
                    rng.shuffle(indices)
                return partition

    raise RuntimeError(
        f"Could not generate a valid partition after {max_attempts} attempts "
        f"(alpha={alpha}, num_clients={num_clients}, "
        f"min_samples_per_client={min_samples_per_client}). "
        "Consider increasing alpha or max_attempts, or lowering the minimum."
    )
