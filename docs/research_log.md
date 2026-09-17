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

## Milestone 3: centralized FedAvg development baseline

**Date:** 2026-09-17

**Purpose:** Establish a conventional centralized federated comparison baseline under the same Dirichlet partitions before decentralized learning is introduced.

Every round gives all clients independent deep copies of the same global model. Each client trains only on its own partition using the existing SGD and epoch-training helpers. The server computes `sum_k (n_k / N) * w_k`, where `n_k` is the client's partition size and `N` is the sum across participating clients. Floating model states are averaged without mutating client inputs; non-floating buffers must agree and are cloned, otherwise aggregation raises an error. Collected client states are kept on CPU; training is sequential on the configured device.

The optimizer is fresh per client per round, preserving momentum only within that local training call. Client training and separate loader generators use `seed + round_number * num_clients + client_id` (one-based rounds). Initialization uses the main seed and partitioning its own local NumPy RNG. Existing cuDNN deterministic settings are retained; cross-hardware or cross-version reproducibility is not guaranteed.

Checks used seed 42, 10 clients, all 60,000 training and 10,000 test samples, one local epoch, three communication rounds, batch size 64, SGD with learning rate 0.01 and momentum 0.9, and `device: auto` resolving to CUDA on the NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch 2.14.0+cu130, CUDA runtime 13.0). Hyperparameters and the dataset were not reduced or tuned between checks.

```bash
python -m tdl.federated.fedavg --config configs/fedavg.yaml --alpha 10.0 --output results/fedavg_alpha_10.json
python -m tdl.federated.fedavg --config configs/fedavg.yaml --alpha 1.0 --output results/fedavg_alpha_1.json
python -m tdl.federated.fedavg --config configs/fedavg.yaml --output results/fedavg_alpha_0.3.json
python -m tdl.federated.fedavg --config configs/fedavg.yaml --output results/fedavg_alpha_0.3_repeat.json
```

All three initial evaluations had test loss 2.3033 and accuracy 9.97%. Actual round metrics were:

| Alpha | Round | Weighted client training loss | Global test loss | Global test accuracy | Clients | Training samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10.0 | 1 | 1.1515 | 0.6485 | 74.86% | 10 | 60,000 |
| 10.0 | 2 | 0.5984 | 0.5131 | 81.09% | 10 | 60,000 |
| 10.0 | 3 | 0.5094 | 0.4631 | 82.86% | 10 | 60,000 |
| 1.0 | 1 | 0.9533 | 0.7603 | 73.27% | 10 | 60,000 |
| 1.0 | 2 | 0.4965 | 0.5340 | 80.03% | 10 | 60,000 |
| 1.0 | 3 | 0.4117 | 0.4736 | 82.68% | 10 | 60,000 |
| 0.3 | 1 | 0.7480 | 1.1799 | 54.61% | 10 | 60,000 |
| 0.3 | 2 | 0.4355 | 0.6519 | 76.64% | 10 | 60,000 |
| 0.3 | 3 | 0.3404 | 0.6120 | 77.03% | 10 | 60,000 |

All clients participated every round with unequal sizes, and the partitions covered every training index exactly once with at least one sample per client. Global accuracy improved across these early rounds for every alpha. Alpha 0.3 completed successfully but had lower global accuracy in this short check; lower local training loss does not by itself imply better global evaluation. The repeated default alpha 0.3 run produced an identical JSON report, including unrounded metrics, on this environment. Generated reports remain local under ignored `results/`.

The full test suite passed (62 tests). Added synthetic tests cover weighted aggregation, input immutability, compatibility and count validation, safe integer buffers, parameter changes during local training, identical client initial weights, global model isolation, tiny full rounds, runner metadata, and reproducibility. The existing centralized training implementation and CNN were not modified.

These are development validation runs, not final scientific results. No decentralized communication, Byzantine behavior, robust aggregation, or Milestone 4 work was implemented.
