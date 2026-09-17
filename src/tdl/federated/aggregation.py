"""Sample-count-weighted averaging of compatible client model states."""

from collections.abc import Mapping, Sequence
from numbers import Integral

import torch


def weighted_average_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    sample_counts: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Return sum(n_k / N * state_k) without mutating inputs.

    Counts must be nonnegative integers with positive total. Floating tensors
    must have matching keys, shapes, dtypes, and devices. Non-floating buffers
    must agree exactly and are cloned rather than numerically averaged.
    """
    if not state_dicts or len(state_dicts) != len(sample_counts):
        raise ValueError("Provide nonempty state_dicts and equally many sample_counts.")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in sample_counts):
        raise ValueError("sample_counts must be nonnegative integers.")
    total = sum(int(n) for n in sample_counts)
    if total <= 0:
        raise ValueError("Total sample count must be positive.")
    reference = state_dicts[0]
    if not reference:
        raise ValueError("Model state dictionaries must not be empty.")
    if any(state.keys() != reference.keys() for state in state_dicts):
        raise ValueError("Model state dictionaries must have matching keys.")
    averaged = {}
    with torch.no_grad():
        for key, tensor in reference.items():
            values = [state[key] for state in state_dicts]
            if any(value.shape != tensor.shape or value.dtype != tensor.dtype
                   or value.device != tensor.device for value in values):
                raise ValueError(f"Incompatible shape, dtype, or device for {key}.")
            if tensor.is_floating_point():
                result = torch.zeros_like(tensor)
                for value, count in zip(values, sample_counts):
                    if count:
                        result.add_(value.detach(), alpha=int(count) / total)
                averaged[key] = result
            else:
                if any(not torch.equal(tensor, value) for value in values):
                    raise ValueError(f"Non-floating buffer {key} differs between clients.")
                averaged[key] = tensor.detach().clone()
    return averaged
