"""Isolated local training and sequential FedAvg round mechanics."""

from copy import deepcopy
from numbers import Integral
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from tdl.federated.aggregation import weighted_average_state_dicts
from tdl.seed import set_seed
from tdl.training.centralized import build_optimizer, train_one_epoch


def train_client(model: nn.Module, loader: DataLoader, config: dict[str, Any],
                 device: torch.device) -> dict[str, Any]:
    """Update a client model with a fresh SGD optimizer and report epoch losses.

    Momentum is retained within this call, but not across communication rounds.
    The caller owns model isolation. All examples must participate each epoch.
    """
    epochs = config["local_epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs <= 0:
        raise ValueError("local_epochs must be a positive integer.")
    if len(loader.dataset) == 0 or loader.drop_last:
        raise ValueError("Client loader must be nonempty and use drop_last=False.")
    model.to(device)
    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    losses = [train_one_epoch(model, loader, criterion, optimizer, device) for _ in range(epochs)]
    return {"num_samples": len(loader.dataset), "epoch_losses": losses,
            "training_loss": sum(losses) / len(losses)}


def run_fedavg_round(global_model: nn.Module, client_loaders: dict[int, DataLoader],
                     config: dict[str, Any], device: torch.device,
                     round_number: int) -> dict[str, Any]:
    """Train every client sequentially, then replace global weights atomically.

    Client IDs must be contiguous from zero. Training and independent loader
    generators are seeded with seed + round_number * num_clients + client_id.
    One-based rounds have distinct seeds; clients do not share mutable loaders
    or optimizer states. States are collected on CPU to limit GPU memory use.
    """
    if not client_loaders or set(client_loaders) != set(range(len(client_loaders))):
        raise ValueError("Provide clients with contiguous IDs starting at zero.")
    if isinstance(round_number, bool) or not isinstance(round_number, Integral) or round_number <= 0:
        raise ValueError("round_number must be a positive integer.")
    states, counts, metrics = [], [], []
    for client_id, loader in sorted(client_loaders.items()):
        client_seed = config["seed"] + round_number * len(client_loaders) + client_id
        set_seed(client_seed)
        if loader.generator is None:
            raise ValueError("Each client loader must have its own seeded generator.")
        loader.generator.manual_seed(client_seed)
        local_model = deepcopy(global_model).to(device)
        local_metrics = train_client(local_model, loader, config, device)
        states.append({key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()})
        counts.append(local_metrics["num_samples"])
        metrics.append({"client_id": client_id, **local_metrics})
        del local_model
    aggregated = weighted_average_state_dicts(states, counts)
    global_model.load_state_dict(aggregated)
    total = sum(counts)
    return {"round": round_number, "num_clients": len(counts), "total_samples": total,
            "weighted_mean_client_training_loss": sum(
                item["training_loss"] * item["num_samples"] / total for item in metrics),
            "clients": metrics}
