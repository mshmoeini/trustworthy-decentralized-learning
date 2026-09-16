# Research log

## Project objective

Build a reproducible PyTorch experimental framework for studying decentralized learning under non-IID client data and Byzantine participants.

**Project started:** 2026-09-14

## Research Question 1

> How does data heterogeneity affect convergence and inter-node agreement in decentralized learning?

Byzantine robustness and Morph-inspired peer selection will be added in later milestones.

## Milestone 1: centralized development baseline

**Date:** 2026-09-14

**Command:**

```bash
python -m tdl.training.centralized --config configs/centralized.yaml
```

The three-epoch development run used CPU and produced the following test-set metrics:

| Epoch | Training loss | Test loss | Test accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 0.5096 | 0.3898 | 85.82% |
| 2 | 0.3064 | 0.3113 | 88.96% |
| 3 | 0.2627 | 0.2699 | 90.14% |

These measurements are a development check for the centralized pipeline, not a final scientific result.

## CUDA development environment verification

**Date:** 2026-09-16

The project `.venv` was verified with Python 3.14.2, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, and the PyTorch CUDA runtime 13.0 on an NVIDIA GeForce RTX 5060 Laptop GPU. The installed NVIDIA driver was 596.13; `nvidia-smi` reported CUDA driver compatibility 13.2.

`torch.cuda.is_available()` returned `True`, with one GPU and current device index 0. A CUDA tensor computation completed successfully. The existing centralized baseline command completed all three epochs with `configs/centralized.yaml` unchanged (`device: auto`), reporting `Device: cuda` and `device: cuda` for every epoch. The full test suite passed (6 tests).

This run verified the local development environment only; it is not a final scientific experiment or a new research milestone. No model, training logic, or dependency constraints were changed.
