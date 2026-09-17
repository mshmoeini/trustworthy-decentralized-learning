"""Model disagreement measured on floating parameters, excluding all buffers."""

from collections.abc import Mapping
from itertools import combinations
import math

import torch
from torch import nn


def mean_pairwise_rms_parameter_distance(models: Mapping[int, nn.Module]) -> float:
    """Average sqrt(sum_p ||theta_i,p - theta_j,p||^2 / P) over i < j.

    P counts all elements of all floating named parameters (including frozen
    parameters), not buffers. Distances are computed on CPU in float64.
    Every unordered pair is weighted equally. One model has distance zero.
    Compatible names/shapes/dtypes and finite parameter values are required.
    """
    if not models:
        raise ValueError("Provide at least one model.")
    parameters = [{key: value for key, value in model.named_parameters() if value.is_floating_point()}
                  for _, model in sorted(models.items())]
    reference = parameters[0]
    total = sum(value.numel() for value in reference.values())
    if not total:
        raise ValueError("Models must have floating parameters.")
    vectors = []
    for state in parameters:
        if state.keys() != reference.keys() or any(
            state[key].shape != value.shape or state[key].dtype != value.dtype
            for key, value in reference.items()
        ):
            raise ValueError("Models must have compatible floating parameters.")
        vector = torch.cat([state[key].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
                            for key in reference])
        if not torch.isfinite(vector).all():
            raise ValueError("Model parameters must be finite.")
        vectors.append(vector)
    distances = [math.sqrt(float((left - right).square().sum()) / total)
                 for left, right in combinations(vectors, 2)]
    return math.fsum(distances) / len(distances) if distances else 0.0
