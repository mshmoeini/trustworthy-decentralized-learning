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

## Milestone 2: reproducible non-IID client partitioning

**Date:** 2026-09-17

**Purpose:** Provide inspectable training-data heterogeneity for future learning experiments, without implementing federated training, peer communication, or attacks.

The utility uses a local NumPy `default_rng(seed)` to shuffle each class's indices, sample symmetric Dirichlet client proportions, and draw multinomial integer counts. Each training index is assigned exactly once; client index lists are also shuffled. Entire allocations below the requested minimum are retried deterministically, with a default bound of 1000 attempts. Impossible minimum-size requests fail immediately. Rejection conditions the allocation on the minimum; client sizes are otherwise unconstrained.

Development checks used all 60,000 Fashion-MNIST **training** samples, 10 clients, seed 42, and a minimum of one sample per client:

```bash
python -m tdl.data.inspect_partition --config configs/partition.yaml --alpha 10.0 --output results/partition_alpha_10.json
python -m tdl.data.inspect_partition --config configs/partition.yaml --alpha 1.0 --output results/partition_alpha_1.json
python -m tdl.data.inspect_partition --config configs/partition.yaml --alpha 0.3 --output results/partition_alpha_0.3.json
```

| Alpha | Minimum client size | Maximum client size | Mean label total variation from global |
| ---: | ---: | ---: | ---: |
| 10.0 | 5,320 | 7,076 | 0.1198 |
| 1.0 | 3,697 | 11,316 | 0.3412 |
| 0.3 | 2,017 | 10,783 | 0.5186 |

Total variation is half the sum of absolute differences between each client's label proportions and the full training set's proportions, averaged equally over nonempty clients. Alpha 10 produced relatively similar label distributions, alpha 1 moderate heterogeneity, and alpha 0.3 visibly stronger skew with some missing client classes. Sample-size ranges need not vary monotonically with alpha. No allocations were tuned to force these observations.

For every alpha, exact coverage and no overlap were checked by comparing all sorted assigned indices to `0..59999`; assigned totals were 60,000, every client met the minimum, and regenerating with the same seed gave identical index lists. Per-client counts and proportions are saved in the ignored local JSON files above. The full test suite passed (39 tests), including synthetic tests of retries and inspection output that do not download data.

These are development validation checks, not final scientific experiments. The existing model, centralized training logic, and CUDA configuration remain unchanged. Milestone 3 has not been started.
