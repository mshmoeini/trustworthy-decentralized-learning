"""Model evaluation utilities."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return mean loss and accuracy for a labeled dataset."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    example_count = 0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            example_count += batch_size

    if example_count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return loss_sum / example_count, correct / example_count
