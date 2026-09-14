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
