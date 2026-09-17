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

## Documentation figures for Milestones 1–3

**Date:** 2026-09-17

Published five documentation figures: the historical CPU centralized baseline, class-proportion heatmaps, client sample counts, per-client label total variation with equally weighted means, and FedAvg global/local learning curves. PNGs support README display and SVGs preserve editable vector output. The centralized figure uses the recorded rounded Milestone 1 table; other plots use the original unrounded development JSON metrics and class counts. No new training run or hyperparameter tuning was performed.

`docs/figure_data/development_checks.json` is a deliberately small archival summary, with source-file SHA-256 hashes, run metadata, and experiment code commit `8fb3cd9d45961305fa4010ec9c49c9fb78337622`. Original partition inspection and FedAvg client counts were checked for agreement, along with the identical repeated alpha 0.3 report. The plotting script validates full class totals, client sizes, configurations, round records, and full participation. The selected snapshot and assets are committed as documentation; raw results, data, caches, and environments remain ignored, with `.gitignore` unchanged.

Reproduction command after installing the optional plotting dependency (`python -m pip install -e ".[plots]"`):

```bash
python scripts/plot_development_checks.py
```

All five figures were visually inspected, all ten PNG/SVG exports reproduced byte-for-byte in this environment (Matplotlib 3.11.2), and all 62 existing tests passed. README captions distinguish clients from independent experimental replications, avoid inferred confidence intervals or significance claims, separate centralized epochs from federated rounds, and distinguish local training loss from global test loss. These remain single-seed development checks, not final research evidence; Milestone 4 has not been started.

## Milestone 4: synchronous multi-topology decentralized development baseline

**Date:** 2026-09-17

Implemented locally: independently owned node CNNs, five seeded communication topologies, synchronous snapshot-based neighborhood aggregation, node-level common-test evaluation, parameter disagreement, and communication metadata. The implementation is recorded in commit `22aeb9ea560a20454a7bee8e37ecc8b672867d6d`.

### Topologies and communication

Graphs are simple, connected, undirected, with node IDs 0–9 and no self-loops. Ring d2 uses modular neighbors ±1; Ring d4 uses ±1 and ±2. Random Regular d4 uses a seeded degree-4 random graph with bounded connectivity retries. Small-World starts from a degree-4 Watts–Strogatz ring, rewires with probability 0.2, and uses the seeded connected generator with bounded retries. Rewiring preserves edge count, not individual degrees. Fully Connected includes every unordered node pair. A local Python RNG with seed 42 generates random graphs independently of training RNGs.

| Topology | Edges | Mean degree | Min degree | Max degree | Transmissions/round | Mean aggregation size including self |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ring d2 | 10 | 2 | 2 | 2 | 20 | 3 |
| Ring d4 | 20 | 4 | 4 | 4 | 40 | 5 |
| Random Regular d4 | 20 | 4 | 4 | 4 | 40 | 5 |
| Small-World d4 | 20 | 4 | 3 | 6 | 40 | 5 |
| Fully Connected | 45 | 9 | 9 | 9 | 90 | 10 |

Traffic is `2 * undirected edges`: idealized directed full-model transmissions per round, excluding self. It does not measure bytes, network latency, or runtime.

### Exact synchronous semantics and mathematics

All nodes begin with independent copies of the same seeded initialization. In each round, every node trains a separate copy of its own round-start model on its own partition. Live round-start states remain untouched throughout local training. All post-training states are cloned into frozen CPU snapshots before aggregation. Each node independently averages self plus its graph neighbors; every next-round state is computed before any live model is updated. There is no central server or persistent global model after initialization.

The baseline is a **sample-count-weighted neighborhood mean**. Let `S_i = {i} ∪ N(i)`, `n_j` be node j's local partition size, and `w̃_j^(t)` its frozen post-training state. The next state is:

`w_i^(t+1) = sum_{j in S_i} [n_j / sum_{k in S_i} n_k] * w̃_j^(t)`.

Floating state entries are averaged; non-floating buffers must agree and are cloned, otherwise aggregation fails. The static baseline uses this rule; Metropolis–Hastings mixing has **not** been implemented. This neighborhood normalization does not imply a globally sample-weighted invariant on an arbitrary graph.

Disagreement after aggregation is the equally weighted mean over all unordered node pairs:

`D = [2 / (K * (K - 1))] * sum_{i<j} sqrt(sum_{p=1}^P (theta_i,p - theta_j,p)^2 / P)`.

`P` counts all elements of floating named parameters, including frozen parameters; all buffers are excluded. Differences are computed on CPU in float64. Node test accuracy and loss means give equal weight to nodes; worst accuracy is the minimum. Accuracy standard deviation is population standard deviation across the ten nodes, not uncertainty across experiment seeds.

### Development configuration and results

All 15 runs used seed 42, 10 nodes, 60,000 Fashion-MNIST training samples, a common 10,000-sample test set, one local epoch per round, three communication rounds, batch size 64, SGD learning rate 0.01 and momentum 0.9, and zero DataLoader workers. Every training index belongs to exactly one partition; partitions match across topologies for each alpha. Initial node models agree exactly. Initialization uses the main seed; partitioning uses its independent seeded NumPy RNG. Node training and independent loader generators use `seed + round_number * num_clients + node_id` with one-based rounds. Optimizers are fresh per node per round; momentum persists only within that local call.

CUDA development environment: Python 3.14.2, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, CUDA runtime 13.0, NetworkX 3.6.1, NVIDIA GeForce RTX 5060 Laptop GPU, driver 596.13 (driver CUDA compatibility 13.2). All recorded runs resolved `device: auto` to CUDA. Deterministic cuDNN settings are enabled; reproduction across hardware/library versions is not guaranteed.

Round-3 results below are formatted from the authoritative local summary; unrounded values are retained in the figure-data snapshot. Std is in percentage points (pp).

| Alpha | Topology | Mean accuracy (%) | Worst accuracy (%) | Accuracy std (pp) | Mean test loss | RMS disagreement |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | Ring d2 | 82.23 | 80.97 | 0.71 | 0.4837 | 0.00124983 |
| 10 | Ring d4 | 82.66 | 81.99 | 0.44 | 0.4707 | 0.00060901 |
| 10 | Random Regular d4 | 82.72 | 82.00 | 0.34 | 0.4705 | 0.00053193 |
| 10 | Small-World d4 | 82.63 | 81.98 | 0.34 | 0.4705 | 0.00062809 |
| 10 | Fully Connected | 82.86 | 82.86 | 0.00 | 0.4631 | 0.00000000 |
| 1 | Ring d2 | 76.70 | 74.04 | 1.89 | 0.5981 | 0.00195631 |
| 1 | Ring d4 | 79.94 | 77.55 | 1.35 | 0.5268 | 0.00095839 |
| 1 | Random Regular d4 | 79.23 | 75.92 | 1.72 | 0.5322 | 0.00088386 |
| 1 | Small-World d4 | 79.82 | 77.19 | 1.54 | 0.5258 | 0.00096172 |
| 1 | Fully Connected | 82.68 | 82.68 | 0.00 | 0.4736 | 0.00000000 |
| 0.3 | Ring d2 | 66.87 | 63.49 | 3.11 | 1.0215 | 0.00259715 |
| 0.3 | Ring d4 | 72.10 | 65.41 | 3.04 | 0.7716 | 0.00134731 |
| 0.3 | Random Regular d4 | 72.87 | 69.02 | 2.36 | 0.7572 | 0.00117404 |
| 0.3 | Small-World d4 | 73.61 | 69.21 | 2.50 | 0.7484 | 0.00141734 |
| 0.3 | Fully Connected | 77.03 | 77.03 | 0.00 | 0.6120 | 0.00000000 |

**Fully Connected/FedAvg sanity check:** The same partitions and training configuration produced exactly equal unrounded weighted training loss, test loss, and accuracy at every round for all three alphas (nine round comparisons). Node accuracy population std and parameter disagreement were exactly zero. This checks the current implementation and weighting rule, without validating arbitrary future mixing rules.

The degree-4 configurations have similar mean accuracies to the fully connected reference at alpha 10 in these checks; gaps increase at alpha 0.3. The three 40-transmission configurations have different observed accuracies and disagreement. These are descriptive **single-seed, three-round, development-only** observations. No confidence intervals, statistical significance, causality, convergence, or universal topology ranking is claimed.

Commands actually used for the existing runs (no retraining for documentation):

```bash
python scripts/run_decentralized_checks.py --alphas 0.3
python scripts/run_decentralized_checks.py --alphas 10.0 1.0
```

Reports: `results/decentralized_{topology}_alpha_{alpha}.json`; consolidated data: `results/milestone4_development_summary.json` and `.csv`. The matrix runner refuses to overwrite per-run reports. Existing reports can be summarized using:

```bash
python scripts/run_decentralized_checks.py --alphas 10.0 1.0 0.3 --summary-only
```

### Documentation and reproducibility

Three new PNG/SVG pairs document actual seeded graphs, round-3 accuracy across the three observed alpha conditions, and idealized communication versus accuracy at alpha 0.3 with numerical disagreement annotations. No fourth trajectory figure was added: three rounds provide limited trajectory information and the README is kept concise.

The existing plotting pipeline now refreshes a deliberately published Milestone 4 snapshot from the local summary and 15 raw reports, verifies summary/raw metric agreement and graph-generator agreement, and checks exact Fully Connected/FedAvg metric equality. SHA-256 hashes identify the summary and reports. Source metadata references implementation commit `22aeb9ea560a20454a7bee8e37ecc8b672867d6d` and hashes its committed Git blob bytes. Original run-time source hashes are retained separately: the recorded runs preceded this commit, and shared helpers were subsequently extracted without changing static training mathematics. The Milestones 1–3 source commit remains `8fb3cd9d45961305fa4010ec9c49c9fb78337622`.

```bash
python scripts/plot_development_checks.py --results-dir results --source-commit 8fb3cd9d45961305fa4010ec9c49c9fb78337622 --milestone4-source-commit 22aeb9ea560a20454a7bee8e37ecc8b672867d6d --verification-date 2026-09-17
python scripts/plot_development_checks.py
```

Normal regeneration reads only the published snapshot and project graph definitions, without raw reports, dataset downloads, or training. Raw reports and datasets remain ignored/local; `.gitignore` is unchanged. All previous five figures remain in the same workflow. The complete suite passed 101 tests for the implemented baseline; documentation validation is recorded after regeneration and visual review.

Documentation validation: all 101 tests passed (6.00 s); all 15 formatted result rows matched the authoritative summary, and all 16 README PNG/SVG paths existed. The two Mermaid blocks were inspected for obvious syntax errors. New PNGs were visually reviewed. SHA-256 checks confirmed the pre-existing implementation, tests, and raw JSON results were unchanged. All eight PNG/SVG pairs regenerated byte-for-byte from the published snapshot with raw-report reading disabled; the five earlier figure pairs matched their committed assets (SVG line endings normalized) in this environment (Matplotlib 3.11.2). During the initial local documentation verification, no commits or pushes were performed.

## EL-Local implementation checkpoint

**Date:** 2026-09-17

Implementation commit `22aeb9ea560a20454a7bee8e37ecc8b672867d6d` also includes EL-Local: independent directed random k-out sampling from all other nodes, uniform averaging of self plus incoming trained snapshots, and retention of the locally trained model when no external model arrives. A separately labelled sample-count-weighted control uses identical sampled graphs. Static baselines retain sample-count weighting, so comparison with uniform EL changes both topology and weighting. The complete suite passed 128 tests. Development checks use one seed (42), three rounds, and ten Fashion-MNIST nodes; no confidence intervals or statistical significance are claimed, and these are not replications of the paper's 100-node experiments. Raw EL reports remain ignored/local; no Epidemic figures are published at this checkpoint.

Documentation checkpoint validation regenerated all eight PNG/SVG pairs without training. Numerical snapshot values were checked against the pre-commit snapshot and remained unchanged. The topology subtitle describes graph construction, and communication-plot disagreement annotations use scientific notation (zero for Fully Connected).
