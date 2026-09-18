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

## Milestone 4.5: Epidemic documentation

**Date:** 2026-09-18

EL-Local provides a dynamic random communication baseline before similarity-guided
Morph. Paper descriptions of regular versus independently sampled communication
are ambiguous; Algorithm 1 and the official `EL_Local` implementation resolve this
project's protocol to directed independent k-out sampling, not a repaired regular
graph. Each sender samples distinct destinations. Every receiver uniformly averages
self plus incoming frozen locally trained models. No incoming models means retaining
its locally trained model. A separate sample-count-weighted control uses identical
graphs. Zero in-degree is possible and tested, but was not observed in these short runs.

| Method | alpha 10 | alpha 1 | alpha 0.3 |
| --- | ---: | ---: | ---: |
| EL-Local uniform k4 | 82.54 | 79.35 | 73.83 |
| EL sample-weighted control k4 | 82.60 | 79.87 | 71.98 |

At alpha 0.3, uniform EL k3/k4/k7 achieved 70.33/73.83/76.36%, using
30/40/70 model transmissions per round; Fully Connected achieved 77.03% at 90.
The matched-budget figure is in the README; the connectivity diagnostic remains
under `docs/figures`. Both have PNG/SVG exports.

All checks use Fashion-MNIST, ten nodes, seed 42, and three rounds. They are
development validation, not statistical evidence. Random dynamic topology alone
has not been shown to improve over static topology: uniform versus weighted EL
at alpha 0.3 (73.83 versus 71.98%) makes aggregation weighting a meaningful confounder.

`scripts/plot_epidemic_checks.py --archive-results results` verifies the existing
summary against raw reports and archives unrounded plot data, report hashes,
configuration, metadata, and committed-source hashes in the existing versioned
snapshot. Provenance references implementation commit `22aeb9ea560a20454a7bee8e37ecc8b672867d6d`.
`python scripts/plot_epidemic_checks.py` regenerates from that snapshot without
training or raw reports. Historical Milestones 1–4 snapshot entries are preserved.
Raw results remain ignored. Part A changes are local and uncommitted.

Validation before Morph: all 128 tests passed. Both new PNG/SVG pairs were visually
reviewed and reproduce byte-for-byte through both plotting entrypoints. All eight
historical figure pairs and all historical snapshot entries remain unchanged.

## Milestone 5A/5B: Morph core and small development check

**Date:** 2026-09-18

Implemented local known-peer views, equal-tensor cosine similarity, five-report
product-based indirect estimates, seeded softmax sampling, directed request
negotiation, configurable refresh/capacity/random slots, and uniform synchronous
snapshot mixing. Exact source reconciliation and deviations are in `docs/morph.md`.
The official revision is `57d73b921c317b0f0d7d9f4a71a6db8051beaf82`.

The paper and released code differ: the code swaps one incoming peer and accepts
all requests; the paper describes full selection, outgoing capacity, and random
exploration. Both selection modes are explicit. The development check uses the
paper-described resampling/capacity path, with exploration restricted to discovered
peers. Incoming request identities and frozen peer-list gossip drive discovery;
there is no global candidate leakage. Random requests share the outgoing cap,
so accepted random counts can be below the requested one slot per receiver.

Added 32 focused tests covering numerical edges, local-view isolation, one-hop
gossip, transitive estimation/history, request discovery, probability/beta behavior,
seed/round reproducibility, incoming/outgoing distinction, capacity displacement,
deficits, refresh intervals, random/guided/hybrid components, official single swaps,
uniform averaging with unequal sample counts, snapshot barriers, execution order,
and a tiny reproducible runner. All 160 tests pass, including the 128 old tests.

After tests passed, ran `python -m tdl.decentralized.morph_runner --config
configs/morph.yaml --output results/morph_alpha_0.3.json`: Fashion-MNIST, 60,000
training/10,000 test samples, ten nodes, seed 42, alpha 0.3, one local epoch, three
rounds, batch 64, SGD lr 0.01/momentum 0.9, CUDA. Morph uses k=4, beta=500,
Delta_r=1, a degree-4 ring bootstrap, outgoing cap=4, and one random proposal
plus three guided proposals per receiver. Delta_r=1 deliberately exercises the
mechanism within three rounds instead of the paper default of five.

| Round | Mean accuracy (%) | Worst (%) | Node std (pp) | Test loss | RMS disagreement | Model transmissions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 49.131 | 42.22 | 5.3399 | 1.450702 | 0.00136789 | 40 |
| 2 | 65.542 | 58.79 | 4.5473 | 0.907971 | 0.00089004 | 40 |
| 3 | 75.204 | 71.82 | 2.1492 | 0.657383 | 0.00069853 | 40 |

Every node maintained incoming/outgoing degree four; all matches filled k. Known
peer counts were four initially, then eight/nine/nine after exchanges. Edge churn
(symmetric difference divided by union) was 0/1/0.974359. Mean selected-peer
similarity was 0.866679/0.942690/0.984704. Accepted random/guided model edges
were 10/30, 5/35, 5/35. Requests were 40/45/54, declines or displacements
0/5/14, and replacements of accepted requests 0/5/6. Model traffic totals 120
transmissions; metadata and request traffic are excluded from that budget.

An initial check exposed a missing requester-identity discovery path. It was fixed
and tested before rerunning the development check. The superseded report is
preserved locally as `results/morph_alpha_0.3_pre_request_discovery_fix.json`; it
is not the validated result. The final report records source/config SHA-256 hashes
and checks source stability throughout the run. Its Git HEAD is explicitly the
base commit, not a claim that the uncommitted Morph implementation is committed.

Validated report SHA-256: `665e94afb9998900c9b99771738d94967427702d1422dddc5c6086a5a21dde50`.

These are single-seed, three-round development mechanics checks, not evidence of
superiority, convergence, or statistical significance. No other alphas were run.
No Byzantine or cyclic-search mechanisms were added. Part A and Morph changes
remain local, with no commit or push. Raw reports remain ignored.

## Milestone 5C: Longer Morph validation and explicit serving modes

**Date:** 2026-09-18

The preceding Milestone 5A/5B entry describes its original checkpoint. Subsequent
three-round capped development checks also completed at alpha=10 and alpha=1,
with final mean accuracy 82.706% and 80.086%. Their report SHA-256 values are
`38f71b1246e69164496f081d670a56f7742833c8c1f67349e6f0bc4b7c393f4d`
and `07b63b3683fe1b85bf3e04cf892568af7921cfde48ce6212110590d3d3005666`.
These short checks tested mechanics and did not establish convergence or superiority.

The planned ten-round alpha=0.3 seed-42 capped Morph check stopped at round 6:
39 edges, receiver8 degree three, all nodes already knowing all nine peers.
Maximum-flow diagnosis found a feasible 40-edge k-by-k assignment, identifying
a negotiation limitation rather than missing candidate discovery. The failed
report remains ignored and unchanged, SHA-256
`1d24bb9bf886f775ab297d7b6c7a21db86997d1b11ab20b47273515061c38f66`.
The separate feasibility analysis is SHA-256
`fdc22aea16d7be02ec95e829ab81145d5156f20993f998877cba858a9d4049a4`.
`tests/fixtures/morph_capped_round6.json` preserves exact preferences/scores and
a 40-edge witness; its regression reproduces 39 and requires an error without
installing an underfilled topology. Maximum flow is diagnostic-only, never an
algorithmic repair. The negotiation remains unchanged and experimental.

Re-checking paper Algorithms 2–3 against official revision
`57d73b921c317b0f0d7d9f4a71a6db8051beaf82` confirmed different serving semantics.
The paper describes outgoing capacity, rejection/cancellation, and replacement
through college-admission-style negotiation. The released `DissDL` execution
maintains `wanted_senders`, communicates request intent, and serves all requesting
receivers without a hard outgoing cap. These paths are now named separately:
primary `morph_code_faithful` and experimental `morph_paper_capped`.

The primary mode retains local discovery, fixed incoming k, uniform aggregation,
and frozen synchronous barriers. No global candidate directory or matching solver
is used. The requested one-random/three-guided `paper_resample` selection remains
an explicit adaptation from the released one-swap/no-separate-random selection;
code-faithful refers to request serving, not full TCP execution replication.
The seven additional mode/regression/end-to-end cases, together with two earlier
observational similarity tests, bring 160 tests to 169. All preceding tests pass.

The uncapped seed-42 ten-round check completed before launching seeds43–46.
All paired runs use Fashion-MNIST, 60,000 training/10,000 test samples, ten nodes,
alpha=0.3, k=4, ten rounds, one local epoch, batch64, SGD lr=0.01/momentum=0.9,
CUDA on RTX5060 Laptop GPU, beta=500, refresh interval=1, and one random plus
three guided Morph slots. Model, optimizer, partition and initial evaluation
controls match between algorithms for each seed. Source stability was checked
throughout every new run. The complete existing EL seed-42 ten-round report was
reused after verifying its controlling sources and paired settings; its historical
Morph module hash refers only to the unchanged observational cosine helper.

| Final metric | Morph mean ± sample std | EL-Local mean ± sample std |
| --- | ---: | ---: |
| Mean accuracy (%) | 81.446 ± 2.780 | 81.707 ± 2.717 |
| Worst-node accuracy (%) | 77.538 ± 2.795 | 75.552 ± 4.758 |
| Node accuracy std (pp) | 2.020 ± 0.297 | 2.902 ± 1.359 |
| Mean test loss | 0.483615 ± 0.051852 | 0.500482 ± 0.069249 |
| RMS disagreement | 0.000489546 ± 0.000058626 | 0.000568021 ± 0.000064597 |

Sample standard deviation uses n−1, n=5; these are not confidence intervals.
Paired Morph-minus-EL mean-accuracy differences for seeds42–46 are
−2.913, +0.522, −0.642, +1.062, +0.667 pp; their mean is −0.2608 pp.
Worst-node differences are +3.300, +3.600, −2.790, −0.770, +6.590 pp,
averaging +1.9860 pp. Morph did not clearly improve mean accuracy. Higher average
worst-node accuracy, lower node spread, lower disagreement and slightly lower
test loss are descriptive observations; no statistical significance is claimed.

Both algorithms delivered exactly 40 models per round, 400 per run, measured from
directed edges and degree sums without imposing a traffic limiter. Fixed four
incoming requests at ten receivers imply 40 model deliveries despite uncapped
outgoing imbalance. Metadata, negotiation and TCP retransmissions are excluded.
Morph outgoing degrees ranged zero–nine; all incoming degrees remained four.
Ten of 50 Morph round graphs lacked strong connectivity, all were weakly
connected. This is an observed topology property, not an underfilled receiver.
EL graphs remained strongly connected. Learning curves fluctuated, including
Morph seed42 round9→10 mean accuracy 81.538→79.614% and an EL seed46 round4
worst-node dip to 53.99% followed by recovery. No nonfinite metric or runtime
failure appeared in the primary validation.

Guided-source instrumentation remains observational. Rounds1–2 are 100% fallback
in every seed, so early behavior is not measured-model-dissimilarity selection.
Round3 pooled proportions are 0.67% fresh direct, 54% cached direct and 45.33%
indirect. Rounds4–10 use direct values; round10 is 70% fresh and 30% cached.
Indirect products may depend on fallback-derived intermediary values. Initial
guided slots, issued guided requests and accepted guided edges agree for this
uncapped mode; each stage is retained separately in the ignored reports.

The published `morph_multiseed` snapshot stores compact unrounded aggregates,
per-run report/config hashes, deduplicated runtime source hashes and base commit
`54fdd31e57b495cb01536a3969858bc06d9678de`. The base is not falsely labelled the
then-uncommitted Morph implementation. Raw artifacts stay ignored. The primary
`morph_vs_epidemic_multiseed.png/.svg` figure shows three final metrics with
across-seed sample-standard-deviation error bars and explicit development labels.
`python scripts/plot_morph_checks.py` regenerates it without raw reports or training;
the full development plotting entrypoint includes it and preserves historical data.
No diagnostic figure was added to keep the README concise. No Byzantine, trust,
robust aggregation or cyclic peer-search implementation was started.

Publication validation: 169 tests passed (6.46s), all 11 PNG/SVG pairs regenerated
byte-for-byte from the versioned snapshot, and the Morph standalone plot matched
the full plotting entrypoint. All preceding raw/support artifacts remained
unchanged. Implementation commit `fc472674f416532c21710e583e6c977d9be5a14c`
contains the Morph core, explicit configurations, source reconciliation and
regression tests. The snapshot records its committed-source hashes separately
from original runtime-byte hashes, preserving line-ending provenance. The
documentation commit publishes compact figure values and assets; raw reports,
caches and temporary files are excluded.


## 2026-09-18 — Byzantine outgoing-update attack milestone

This milestone implements attacks and observational reporting only. No trust,
validators, clipping, robust aggregation, anomaly detection, or attack-aware peer
selection is added. The experiment base is
`cb71407cc388f1838a14638441ab0f3abd85a935`; runs preceded the attack implementation
commit. The compact Byzantine figure snapshot preserves original runtime hashes
separately from committed Git blob hashes.

### Architecture and update definitions

Capture the configured attacker's round-start state before normal local training.
Train every node normally and freeze all post-training states before transforming
outgoing payloads. For `delta = theta_local - theta_start`:

- Sign flip sends `theta_start - lambda * delta` (validated lambda=1).
- Update scaling sends `theta_start + lambda * delta` (validated lambda=5).
  The configuration/CLI name is `scaling`; summary labels use `update_scaling`.

Only outgoing payloads are changed. Inputs and returned tensors have independent
ownership; nonfloating buffers keep their local values. Aggregation uses the
unchanged local post-training state for self and outgoing payloads for incoming
senders. All next-round aggregates are computed before any model is installed.
The same payload is sent to every requesting receiver; this is neither
equivocation nor data poisoning or metadata forgery. Honest models can be affected
by later aggregation of poisoned incoming models.

Morph scores its honest current local state against previously received payloads
and cached/indirect estimates; the current payload cannot influence current-round
selection before it is received. Candidate diagnostics observe existing local
scores without rescoring, consuming RNG, or using attacker IDs in the selection
algorithm. Rank bounds retain ties, with lower similarity ranked first. Initial
fallback/indirect scores are distinguished from fresh or cached direct scores.
First/top-2/top-3 fractions require rank_max <= k for guaranteed membership; possible
fractions use rank_min <= k. All fractions use all nine honest receivers, with
unknown candidates unranked. A fallback tie does not demonstrate measured diversity.

Configuration validation rejects invalid identities, strengths and incompatible
states; overflow is reported without clipping. The dedicated runner records
configuration, partition, source hashes, initial evaluation, learning curves,
actual traffic, poisoned deliveries, local ranks, and paired clean differences.
It requires an exact completed clean control for the CLI, checks all non-attack
configuration fields and shared training/model/data sources, and refuses to
overwrite reports. No-attack equivalence, snapshot timing, honest self state,
payload ownership, invalid inputs, ranks and deterministic matched-control runs
are covered by the 25 added tests (194 total).

### Single-seed development and five-seed validation

Seed 42 development used ten rounds at alpha 0.3. Honest mean accuracy for Morph
was 79.471% clean, 75.870% sign flip and 79.427% scaling; attacker-origin deliveries
were 31, 37 and 78 out of 400. EL honest means were 82.281%, 79.814% and 81.399%,
with 40 origin deliveries in each case. This illustrates why attacker selection
frequency and learning damage must be evaluated separately: seed 42 scaling nearly
doubled Morph's attacker-origin share while final honest mean damage was 0.044 pp.

The primary matrix is Fashion-MNIST, ten nodes, alpha 0.3, ten rounds, one local
epoch, seeds 42–46, fixed Byzantine node 0, EL-Local k=4 uniform and Morph
code-faithful k=4 uniform. SGD learning rate 0.01, momentum 0.9 and batch size 64,
as well as all data/model settings, remain unchanged. Morph beta 500, refresh 1,
one random and three guided slots, uncapped outgoing service remain unchanged.
Twenty-one new runs comprise sixteen seed 43–46 attack cases and five clean Morph
replays for candidate diagnostics. Five EL clean and four seed 42 attack reports
were reused after validation. Every clean Morph replay exactly matches the
original round metrics, node evaluations and topology.

Honest-only metrics exclude node 0 in both clean and attack runs; each attack is
paired with its own algorithm/seed clean control. Global mean/worst/spread, mean
test loss, RMS disagreement and attacker accuracy are retained separately.
The following final-round changes are **attack minus clean**, in percentage
points, with across-seed **sample** standard deviation (n-1, n=5):

| Algorithm / attack | Honest mean change | Honest worst change | Honest std change |
| --- | ---: | ---: | ---: |
| EL sign flip | −1.686 ± 1.156 | −2.176 ± 1.041 | +0.285 ± 0.501 |
| EL scaling | −2.174 ± 1.996 | −3.012 ± 4.176 | +1.091 ± 1.346 |
| Morph sign flip | −2.149 ± 1.489 | −2.116 ± 2.472 | −0.000126 ± 0.546 |
| Morph scaling | −6.426 ± 4.298 | −8.656 ± 8.838 | +1.241 ± 2.678 |

Morph scaling honest-mean changes by seed 42–46 are −0.044, −8.053, −4.100,
−9.856 and −10.078 pp; effects vary substantially. Scaling seed 46 worst-node
change is −23.14 pp. EL scaling slightly improves honest mean in seed 43 (+0.24 pp)
and worst accuracy in seed 42 (+3.71 pp). Final disagreement rises in every attacked
run: mean increases EL sign flip 0.00003834, EL scaling 0.00034610, Morph sign
flip 0.00006848 and Morph scaling 0.00029196. Scaling curves oscillate markedly,
including Morph seed 44 honest accuracy 51.891% at round 9 and 79.040% at round 10.

### Selection amplification and poisoned deliveries

Attacker-origin count includes clean node 0 sends; poisoned-delivery count is zero
for clean and equals attacker-origin count for the validated attacks. There are
400 model deliveries per run. EL remains 40/400 (10%) and its graphs match clean
for every seed. Morph origin shares average 8.40% clean, 10.10% sign flip and 18.45%
scaling (sample std 2.97, 3.05, 1.01 percentage points, respectively).

| Seed | Morph clean origin count | Sign flip count / poisoned share | Scaling count / poisoned share | Guided amplification sign / scaling |
| --- | ---: | ---: | ---: | ---: |
| 42 | 31 | 37 / 9.25% | 78 / 19.50% | +6 / +47 |
| 43 | 21 | 32 / 8.00% | 68 / 17.00% | +11 / +47 |
| 44 | 53 | 62 / 15.50% | 77 / 19.25% | +9 / +24 |
| 45 | 34 | 35 / 8.75% | 74 / 18.50% | +1 / +40 |
| 46 | 29 | 36 / 9.00% | 72 / 18.00% | +7 / +43 |

All per-seed origin-count increases are guided; random attacker deliveries match
clean. Guided proposals equal accepted selections in these uncapped cases.
Mean guided amplification is +6.8 ± 3.77 sign flip and +40.2 ± 9.52 scaling.
Mean attacker outgoing degree across seeds is 3.36 clean, 4.04 sign flip and 7.38
scaling; incoming degree remains 4 in every round. Scaling ranks the attacker
uniquely first for all nine honest receivers in seed 42 rounds 7, 8, 10; seed 44
rounds 6–10; seed 45 rounds 7, 9; seed 46 round 10; none for seed 43. Sign flip reaches
this condition only in seed 44 rounds 7–8. Clean never reaches it. Rank membership
need not coincide with acceptance due to random reservations and candidate scores.

### Descriptive association and caveats

RQ-A honest learning damage, RQ-B selection amplification and RQ-C association are
evaluated separately. Positive damage here means clean minus attacked accuracy;
selection amplification means attacked minus clean origin-delivery count.
Pearson / Spearman diagnostics are:

| Morph group | Honest mean damage | Honest worst damage |
| --- | ---: | ---: |
| Sign flip, n=5 | −0.352 / −0.500 | +0.048 / +0.200 |
| Scaling, n=5 | +0.091 / −0.205 | +0.325 / +0.154 |
| Pooled attacks, n=10 | +0.567 / +0.280 | +0.551 / +0.535 |

Clean zero anchors are excluded. The pooled ten runs repeat the same five seeds
and can be confounded by attack type; they are not ten independent seeds.
Within-attack associations are weak or inconsistent. No p-values, statistical
significance, causal effect, convergence or general robustness/vulnerability claim
is made. Selection amplification, attack payload severity, and learning damage
are distinct phenomena. Sign flip reverses the update; scaling enlarges it in its
original direction. In the current five-seed development validation, scaling
consistently increases Morph attacker selection and is associated with larger
average honest-node degradation than under EL. This does not imply that scaling
always breaks Morph or that selection alone causes damage. No strength sweep was
run and no monotonic strength-response claim is supported.

All 300 round cases have 40 directed full-model transmissions (400 per run),
excluding metadata/negotiation. No runtime or nonfinite-metric failure, declined
request or underfilled Morph receiver appeared. All graphs are weakly connected;
EL remains strongly connected. Morph lacks strong connectivity in=10/50 clean,
9/50 sign-flip and 11/50 scaling round graphs. These are observed topology
properties, retained without repairs. Outgoing imbalance and learning oscillations
are reported separately from execution failures.

### Publication and reproducibility

The compact `docs/figure_data/byzantine_validation.json` publishes only per-seed
values needed for two figures plus setup/provenance. Raw learning curves, ranks,
reports, configurations, logs and all 76 multi-seed artifacts remain ignored.
`python scripts/plot_byzantine_checks.py` regenerates both PNG/SVG pairs from
versioned data without training or reading raw results. The damage figure uses
paired honest-mean changes, a visible zero baseline and sample-standard-deviation
error bars. The selection figure uses attacker-origin shares, individual seed
points and sample-standard-deviation bars; clean sends are not labelled poisoned.
Both carry explicit five-seed development captions and no significance/causality
claims. Tests pass 194 before milestone commits; final tests and figure
reproducibility are checked again before pushing only develop.

Implementation commit `0dfa83701e03e8cce828f7e97b1edce3a7006b59` contains only attack code, configurations,
communication hooks and tests. Publication checks passed 194 tests and both
PNG/SVG pairs reproduced byte-for-byte from the compact snapshot. The separate
documentation commit publishes these figures and notes; raw reports are excluded.
