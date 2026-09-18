"""Transform sent updates while preserving independently owned honest states."""
import math
from numbers import Integral, Real

import torch

ATTACKS = ("none", "sign_flip", "scaling")


def validate_attack_config(config, node_ids):
    name = config.get("attack", "none")
    strength = config.get("attack_strength", 1.)
    nodes = config.get("byzantine_nodes", [])
    if name not in ATTACKS:
        raise ValueError("Unknown attack")
    if isinstance(strength, bool) or not isinstance(strength, Real) or not math.isfinite(strength) or strength < 0:
        raise ValueError("attack_strength must be finite and nonnegative")
    if not isinstance(nodes, (list, tuple)) or any(isinstance(n, bool) or not isinstance(n, Integral) for n in nodes):
        raise ValueError("byzantine_nodes must be explicit integer IDs")
    if len(set(nodes)) != len(nodes) or not set(nodes) <= set(node_ids):
        raise ValueError("Duplicate or unavailable Byzantine IDs")
    return name, float(strength), tuple(sorted(nodes))


def apply_attack(attack_name, theta_start, theta_local, attack_strength=1., rng=None):
    """Return owned payload: start + factor * (local - start).

    factor is -lambda for sign_flip, +lambda for scaling; none clones local
    without arithmetic. Nonfloating buffers retain their local value. No RNG is
    consumed by these deterministic attacks. Neither input is ever mutated.
    """
    name, strength, _ = validate_attack_config(
        {"attack": attack_name, "attack_strength": attack_strength}, [])
    if not theta_local or set(theta_start) != set(theta_local):
        raise ValueError("States must have the same nonempty keys")
    payload = {}
    for key, local in theta_local.items():
        start = theta_start[key]
        if local.shape != start.shape or local.dtype != start.dtype or local.device != start.device:
            raise ValueError(f"Incompatible state tensor: {key}")
        if local.is_complex():
            raise ValueError("Complex states are unsupported")
        if local.is_floating_point():
            if not torch.isfinite(local).all() or not torch.isfinite(start).all():
                raise ValueError("Attack inputs must be finite")
            factor = -strength if name == "sign_flip" else strength
            value = local.detach().clone() if name == "none" else start.detach() + factor * (local.detach() - start.detach())
            if not torch.isfinite(value).all():
                raise ValueError("Attack payload overflowed; no clipping is applied")
            payload[key] = value
        else:
            payload[key] = local.detach().clone()
    return payload


def start_snapshots(models, config):
    """Capture pre-training weights only when a configured attack is active."""
    name, _, nodes = validate_attack_config(config, models)
    if name == "none" or not nodes:
        return None
    return {n: {k: v.detach().cpu().clone() for k, v in models[n].state_dict().items()} for n in nodes}


def outgoing_payloads(theta_start, theta_local, config):
    """Shared sender boundary; unconfigured senders remain honest, with no alias."""
    name, strength, nodes = validate_attack_config(config, theta_local)
    active = set(nodes) if name != "none" else set()
    if active and (theta_start is None or not active <= set(theta_start)):
        raise ValueError("Active attackers require pre-training snapshots")
    return {n: apply_attack(name, theta_start[n], state, strength) if n in active
            else {k: v.detach().clone() for k, v in state.items()}
            for n, state in sorted(theta_local.items())}
