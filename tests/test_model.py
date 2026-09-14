"""Tests for the Fashion-MNIST model."""

import torch

from tdl.models import FashionMNISTCNN


def test_model_output_shape() -> None:
    model = FashionMNISTCNN()
    inputs = torch.randn(8, 1, 28, 28)

    logits = model(inputs)

    assert logits.shape == (8, 10)
