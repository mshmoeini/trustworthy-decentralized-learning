"""Centralized Fashion-MNIST training entry point."""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from tdl.config import load_config
from tdl.data.fashion_mnist import load_fashion_mnist
from tdl.models import FashionMNISTCNN
from tdl.seed import set_seed
from tdl.training.evaluate import evaluate


def resolve_device(requested: str) -> torch.device:
    """Resolve an automatic or explicit PyTorch device request."""
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(normalized)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> Optimizer:
    """Construct the configured optimizer for the model."""
    optimizer_name = str(config["optimizer"])
    if optimizer_name.lower() != "sgd":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return SGD(
        model.parameters(),
        lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean example loss."""
    model.train()
    loss_sum = 0.0
    example_count = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        example_count += batch_size

    if example_count == 0:
        raise ValueError("Cannot train on an empty dataset.")
    return loss_sum / example_count


def run(config_path: str | Path) -> None:
    """Run centralized training from a YAML configuration file."""
    config = load_config(config_path)
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))

    train_dataset, test_dataset = load_fashion_mnist(config["data_dir"])
    loader_generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **loader_options,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = FashionMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    epochs = int(config["epochs"])

    print(f"Device: {device}")
    for epoch in range(1, epochs + 1):
        training_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}/{epochs} | training loss: {training_loss:.4f} | "
            f"test loss: {test_loss:.4f} | test accuracy: {test_accuracy:.2%} | "
            f"device: {device}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML training configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line training program."""
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
