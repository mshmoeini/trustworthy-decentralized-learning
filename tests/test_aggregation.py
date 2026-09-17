"""Cheap tests for weighted model aggregation and buffer policy."""

import pytest
import torch

from tdl.federated.aggregation import weighted_average_state_dicts


def test_weighted_math_and_input_isolation() -> None:
    a = {"weight": torch.tensor([1., 1.]), "counter": torch.tensor(7)}
    b = {"weight": torch.tensor([3., 3.]), "counter": torch.tensor(7)}
    result = weighted_average_state_dicts([a, b], [1, 3])
    assert list(result) == list(a)
    assert torch.equal(result["weight"], torch.tensor([2.5, 2.5]))
    assert result["counter"].dtype == torch.int64
    result["weight"].zero_()
    result["counter"].zero_()
    assert torch.equal(a["weight"], torch.ones(2))
    assert torch.equal(b["weight"], torch.full((2,), 3.))
    assert a["counter"].item() == b["counter"].item() == 7


@pytest.mark.parametrize("states,counts", [([], []), ([{}], [1]),
    ([{"w": torch.ones(1)}], []), ([{"w": torch.ones(1)}], [0]),
    ([{"w": torch.ones(1)}], [-1]), ([{"w": torch.ones(1)}], [1.5]),
    ([{"w": torch.ones(1)}], [True])])
def test_invalid_inputs(states, counts) -> None:
    with pytest.raises(ValueError):
        weighted_average_state_dicts(states, counts)


@pytest.mark.parametrize("other", [{"x": torch.ones(2)}, {"w": torch.ones(3)},
    {"w": torch.ones(2, dtype=torch.float64)}])
def test_incompatible_states(other) -> None:
    with pytest.raises(ValueError):
        weighted_average_state_dicts([{"w": torch.ones(2)}, other], [1, 1])


def test_differing_integer_buffers_are_rejected() -> None:
    with pytest.raises(ValueError, match="Non-floating buffer"):
        weighted_average_state_dicts([{"counter": torch.tensor(1)},
                                     {"counter": torch.tensor(2)}], [1, 1])


def test_zero_weight_and_single_client() -> None:
    a, b = {"w": torch.tensor([1.])}, {"w": torch.tensor([3.])}
    assert torch.equal(weighted_average_state_dicts([a, b], [0, 3])["w"], b["w"])
    assert torch.equal(weighted_average_state_dicts([a], [2])["w"], a["w"])
