# Morph implementation and source reconciliation

## Validated primary baseline

The primary baseline is `morph_code_faithful`, with uncapped request serving and
the documented three-guided/one-random `paper_resample` selection adaptation.
Ten-round validation used Fashion-MNIST, ten nodes, alpha=0.3, seeds 42–46,
one local epoch, k=4, beta=500, refresh interval=1, and uniform aggregation.
Optimizer, model and data settings were unchanged. The complete suite passes
169 tests, including the unchanged capped-failure regression.

| Final metric | Morph | EL-Local uniform |
| --- | ---: | ---: |
| Mean accuracy (%) | 81.446 ± 2.780 | 81.707 ± 2.717 |
| Worst-node accuracy (%) | 77.538 ± 2.795 | 75.552 ± 4.758 |
| Node accuracy std (pp) | 2.020 ± 0.297 | 2.902 ± 1.359 |
| Mean test loss | 0.483615 ± 0.051852 | 0.500482 ± 0.069249 |
| RMS disagreement | 0.000489546 ± 0.000058626 | 0.000568021 ± 0.000064597 |

Values are across-seed means ± sample standard deviations, not confidence
intervals. Mean Morph-minus-EL accuracy was −0.2608 pp; worst-node accuracy was
+1.9860 pp. Morph did not clearly improve mean accuracy in this setup. Higher
average worst-node accuracy, lower node spread, lower disagreement and slightly
lower test loss are descriptive observations, without significance claims.

Actual communication was 40 model deliveries per round and 400 per run for both
methods. Morph incoming degree remained four; outgoing degree ranged from zero
to nine across runs. Ten of 50 Morph round graphs were not strongly connected;
all remained weakly connected. This is an observed topology property, not an
underfilled receiver or runtime failure. Learning curves fluctuated.

| Round | Fresh direct (%) | Cached direct (%) | Indirect (%) | Fallback (%) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 100 |
| 2 | 0 | 0 | 0 | 100 |
| 3 | 0.67 | 54.00 | 45.33 | 0 |
| 4 | 8.00 | 92.00 | 0 | 0 |
| 5 | 29.33 | 70.67 | 0 | 0 |
| 6 | 45.33 | 54.67 | 0 | 0 |
| 7 | 64.67 | 35.33 | 0 | 0 |
| 8 | 69.33 | 30.67 | 0 | 0 |
| 9 | 73.33 | 26.67 | 0 | 0 |
| 10 | 70.00 | 30.00 | 0 | 0 |

Fractions pool 150 guided selections per round across five seeds. Initial slots,
issued requests and accepted guided edges agree in this uncapped mode. Rounds
1–2 are entirely fallback-driven: early behavior must not be described as driven
by measured model dissimilarity. Indirect estimates can themselves depend on
fallback-derived intermediary scores; cached direct scores are not fresh
comparisons against the current local model. Similarity logic was not changed.

Compact aggregates and report/config/runtime-source SHA-256 provenance are in
`docs/figure_data/development_checks.json`, under `morph_multiseed`. Raw reports
remain ignored. Regenerate the primary PNG/SVG pair without raw reports or training:

```bash
python scripts/plot_morph_checks.py
```

`python scripts/plot_development_checks.py` includes the Morph figure with the
historical figures. The research log records the three-round development phase,
round-6 capped failure, mode separation, and longer validation provenance.

Sources inspected: [paper, equations 3–6 and Algorithms 2–3](https://arxiv.org/html/2602.03383v1)
and [official code at 57d73b9](https://github.com/bacox/Morph/tree/57d73b921c317b0f0d7d9f4a71a6db8051beaf82).
Relevant functions are `DissDL.compute_similarity`, `estimate_similarity`,
`update_wanted_senders`, `_handle_syn`, `_handle_syn_ack`, `_handle_ack`, `run`,
and `PlainAverageSharing._averaging`. `GlobalModelSimilarity` duplicates these
mechanisms but also computes a global-model diagnostic; that global diagnostic
is deliberately absent here.

## Explicit protocol modes

`morph_code_faithful` is the primary baseline. Each receiver maintains exactly k
wanted senders selected from its own discovered view. Request intents define
directed sender→receiver delivery, and each sender serves **all** requesters.
There is no outgoing cap, receiver ranking, capacity rejection, matching solver,
or global candidate search. `outgoing_capacity` must be null. Wanted senders persist
between topology refreshes. The simulator assembles a delivery graph from local
requests; this bookkeeping cannot add or choose peers. Uniform aggregation and
frozen synchronous barriers are unchanged. Default runner configuration is
`configs/morph_code_faithful.yaml`.

`morph_paper_capped` preserves the existing experimental receiver-proposing
negotiation unchanged. The paper attempts an outgoing cap k, describes replacing
less dissimilar accepted requests, notifying rejection/cancellation, and searching
again in a college-admission-style procedure, while targeting fixed incoming k.
The pinned `DissDL.run` instead communicates `wanted_senders` through request
intent and sends to `nodes_requesting_from_me | partial_connections`; its SYN
handlers do not enforce a hard outgoing cap. These are separate execution modes,
not one hybrid capacity policy. The experimental capped mode remains unsuitable
as the primary baseline: the seed-42 round-6 case negotiates only 39 edges even
though all nodes know all nine peers and a feasible 40-edge assignment exists.
`tests/fixtures/morph_capped_round6.json` retains exact preferences/scores and a
feasibility witness. Regression tests replay the deficiency and require a raised
error, not silent underfilling. Maximum flow remains diagnostic-only; neither
mode calls it. No capped repair has been implemented in this task.

Protocol mode and selection profile are independent and explicitly recorded.
The requested validation retains `selection_mode: paper_resample`, one random
slot plus three guided slots, beta=500, Delta_r=1, and k=4. This is code-faithful
**request serving** with the previously documented paper-hybrid selection
adaptation. It is not a claim to reproduce the released code's one-swap selection
or TCP timing. The released selection path is separately available as
`selection_mode: official_swap`, `random_peer_count: 0`.

Communication is counted from actual directed delivery edges. With k distinct
incoming requests at every node and one model sent per request, `sum(in_degree)
= n*k`; uncapped outgoing degree can be unbalanced without increasing this total.
This is a consequence of the observed requests, not a communication limiter.
Transport retransmissions/handshake model messages are not emulated; actual
degree distributions and edge counts are saved for every round.

Similarity-source instrumentation remains observational: initial guided slots,
issued guided requests, and accepted guided edges are separate stages. Four-way
fractions expose fresh direct, cached direct, indirect, and fallback scores. The
old capped run's rounds 1–2 were entirely fallback-driven; they must not be
interpreted as measured model-diversity selection. Source logic is unchanged.

## Similarity and discovery

Each node initially knows only the predecessors and successors of its bootstrap
overlay. The simulator's node registry is used for delivering models, never for
constructing candidate lists. Models carry the sender's known-node list and
nonzero cached similarities with known nodes. Lists expand local discovery;
there is no global refresh or discovery oracle. Metadata is frozen before delivery,
so discovery cannot transit more than one hop in an exchange.
Receiving a connection request also reveals the requester's identity to its
sender, mirroring the code's request-intent ingestion. These discoveries affect
the next selection opportunity, never the already-frozen current preference lists.

Similarity is the arithmetic mean of cosine similarities of flattened named
parameter tensors. Biases are separate tensors; buffers are excluded. Zero-norm
pairs are skipped, with -1 returned when all pairs are unusable, as in official
`compute_similarity`. The project rejects incompatible/nonfinite tensors rather
than silently skipping mismatched keys/shapes. Float64 normalized computations
and clipping to [-1,1] protect numerical edge cases.

For an indirectly observed peer z, retain the five most recently delivered
`(round, intermediary y, reported sim(y,z))` items. Estimate `sim(i,z)` by the
arithmetic mean of `cached sim(i,y) * reported sim(y,z)`. This is the code's product
estimator, not an invented angular-bound estimator. Cached intermediaries fall
back to the mean cached similarity (zero for an empty cache). Directly observed
peers use the latest stored model, or their cached value if those weights were
discarded. First direct receipt clears indirect history. Timestamps order retained
reports; there is no age decay or TTL in the official estimator.

Official report ingestion is gated by zero-based `iteration > 0` and
`iteration % (Delta_r - 1) == 0`. The project preserves that gate after converting
to one-based rounds, but uses `max(1, Delta_r - 1)` to repair the division-by-zero
bug when Delta_r=1. No stale/future transport messages exist in this synchronous
simulation. Selection uses the locally trained model against previously received
peer snapshots; the current global model registry cannot supply unseen models.

Softmax probabilities are proportional to `exp(-beta * similarity)` for additions,
sampled stochastically without replacement. Positive beta favors dissimilarity;
beta=0 permits a uniform ablation. Subtracting the maximum logit prevents overflow.
Sorted candidates and per-receiver, per-round seeded RNGs avoid set-order dependence.

Uniform aggregation is `(self + sum(incoming models)) / (1 + in_degree)`.
Directed edge sender→receiver means one model transmission. All post-training
models are frozen before selection/exchange, metadata is frozen before gossip,
and every aggregate is computed before any live model is installed. Existing
safe nonfloating-buffer policy is retained; the current CNN has no such buffers.

## Paper/code differences exposed explicitly

The pinned code's `update_wanted_senders` replaces **one** incoming peer per
refresh: add via `softmax(-beta*sim)`, remove via `softmax(+beta*sim)`. It skips
refresh at iteration zero, cannot swap when there is only one current sender,
and retains the others. `selection_mode: official_swap`, `random_peer_count: 0`,
and `outgoing_capacity: null` reproduce this selection/request behavior in the
synchronous architecture. The project evaluates refresh on one-based
`round % Delta_r == 0`, as paper Algorithm 2 specifies, so refresh timing differs
from the code's zero-based loop. It does not emulate TCP handshake delays.

The code accepts every SYN and sends models to every requester. There is no
outgoing-capacity rejection, stable matching, or separate random injection in
that path. Incoming degree is a target maintained by swaps; outgoing degree
can vary. This discrepancy is not hidden behind a claim of exact code replication.

`selection_mode: paper_resample` implements the paper's sequential softmax
selection. `outgoing_capacity: k` enables its receiver-proposing negotiation:
senders accept up to capacity, rank requests using the similarity reported by
the receiver, replace a less dissimilar accepted requester, and notify rejection
or displacement through the matching state. Rejected receivers continue through
their local stochastic preference lists. Equal scores use node ID as a stable tie
break. Each receiver proposes to each discovered sender at most once, ensuring
termination without relying on the paper's stated step bound. If local discovery
cannot fill k, the simulator raises a diagnostic error instead of silently using
global candidates or claiming a fixed degree. No scientific experiment proceeds
from a deficient topology.

The paper adds random peers to guided peers. Its Algorithm 3 distinguishes a full
candidate set C and a local scored set C_A, but the released code has no such
random component. This project's explicit adaptation samples random peers from
the discovered local view, reserves those slots before guided sampling, and uses
the remaining discovered peers for guided selection. It never introduces unknown
IDs through a global C. Random requests can be declined under the same capacity
rule; metrics report accepted random edges separately. Hybrid mode does not
guarantee connectivity. `random_peer_count=k` is the random-only ablation; zero
is guided-only. Hybrid mode requires `paper_resample`.

The original capped development config (`configs/morph.yaml`) uses ten Fashion-MNIST nodes, a degree-4 ring bootstrap,
target incoming k=4, outgoing capacity=4, one random slot plus three guided slots,
beta=500, and Delta_r=1 so a three-round check actually exercises topology changes.
These are development settings, not the paper's 100-node/8000-iteration experiment
or its default Delta_r=5. Training uses the existing full-epoch CNN workflow and
fresh per-round SGD optimizers. Model traffic excludes metadata and negotiation.
