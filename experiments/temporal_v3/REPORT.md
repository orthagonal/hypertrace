# Temporal Memory v3 — measured held-out report

Frozen validation choice: **w8_best**, dependency candidate budget **16/16**.
Validation window recall@2: **69.6%**. The complete
selection record is in frozen_settings.json. Freeze SHA-256: `4ac1b2dc251081214e7548647dbc8baa8e282bf086c1336ab4ac87dafa1c8d2e`.
Validation used seed 2027; held-out evaluation used seed 42, with independent filler
replicas. The freeze binds retrieval code/metrics and validation data hashes. It was
checked before test retrieval; no setting is selected from the following test results.

At 32,768 dense-target actions, window recall@2 is **12.5%**
and dependency-verified recall@2 is **100.0%**,
versus 0% for the existing v2 object-scoped experiment on this corpus.
These results concern a finite synthetic query task, not a new classifier or enforcer.
Random-ranking recall@2 has expectation 12.5% in this gallery. **Window-only recall in the longest dense-target case is at or below the random-ranking expectation; this does not solve the dense-target failure reliably.**
With budget 16/16, exact verification checks the whole gallery; the same accepted set
can be obtained by exhaustive raw-graph matching without HDC shortlisting.

## Retrieval comparison

Cells: **recall@1 / recall@2 (%)**. Each collection preserves v2's 16 gallery histories
and eight held-out queries, with two relevant candidates per query. Thus recall@1 has
a 50% ceiling. Relevance is the same motif family **and order**, not the outcome label.
All original histories, v1 trees and v2 anchor representations remain unchanged.

| Profile | Actions | v1 | v2 global | v2 scoped | v3 windows | v3 + dependencies | Exact baseline |
|---|---|---|---|---|---|---|---|
| sparse | 32 | 25.0/31.2 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 |
| dense_unrelated | 32 | 0.0/6.2 | 25.0/31.2 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 |
| dense_target | 32 | 12.5/25.0 | 25.0/31.2 | 25.0/31.2 | 0.0/18.8 | 50.0/100.0 | 50.0/100.0 |
| sparse | 1024 | 18.8/31.2 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 |
| dense_unrelated | 1024 | 6.2/12.5 | 6.2/6.2 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 |
| dense_target | 1024 | 0.0/0.0 | 0.0/0.0 | 6.2/12.5 | 6.2/31.2 | 50.0/100.0 | 50.0/100.0 |
| sparse | 4096 | 12.5/18.8 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 |
| dense_unrelated | 4096 | 12.5/25.0 | 12.5/25.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 |
| dense_target | 4096 | 6.2/6.2 | 0.0/18.8 | 0.0/0.0 | 12.5/25.0 | 50.0/100.0 | 50.0/100.0 |
| sparse | 32768 | 18.8/37.5 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 |
| dense_unrelated | 32768 | 6.2/12.5 | 0.0/0.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 | 50.0/100.0 |
| dense_target | 32768 | 12.5/25.0 | 0.0/0.0 | 0.0/0.0 | 12.5/12.5 | 50.0/100.0 | 50.0/100.0 |
| dense_target_far | 4096 | 0.0/0.0 | 12.5/18.8 | 6.2/12.5 | 0.0/0.0 | 50.0/100.0 | 50.0/100.0 |
| dense_target_far | 32768 | 0.0/0.0 | 12.5/25.0 | 6.2/12.5 | 6.2/12.5 | 50.0/100.0 | 50.0/100.0 |

## What was searched and verified

All 8/16/32-anchor windows, 50% overlap, partial windows and complete tail contexts
are stored and individually searchable. GPU compares the four query-context slots
against **every stored window** and returns full-gallery window top-K. History ranking
uses eight deterministic, equally budgeted temporal contexts per candidate to avoid
giving longer histories more minimum-distance opportunities. No stored history/window
is truncated; bounded context scoring is a query policy, not exhaustive history aggregation.

Validation compared best-window, normalized top-two-window mean, and boundary-preserving
ordered multi-window alignment, for all three widths. Full predeclared test ablations
are in query_metrics.csv; the chosen method above remains the validation choice.

The optional CPU exact stage follows raw backward dependencies from the first/last
observed scoped anchors, after HDC candidate proposal. It matches typed event DAGs,
writer-version aliases and mapped chronology; no precomputed answer signature, oracle
outcome or future event enters that stage. It can abstain and cannot recover a relevant
history outside its proposed candidate budget. Candidate recall and abstention are
reported separately. If the selected budget equals 16, it verifies the entire gallery;
that success must not be credited to HDC shortlisting alone.

Far cases place 256 or 2,048 **other significant same-object events** between the
critical operations. Selected exact links span up to 24,575 observed steps on
object 2; selected_dependency_links.csv resolves both endpoints to stored windows.
The verifier has no hop cutoff; a 32,768-node chain regression passes.

## Incorrect matches, abstention and ties

Cells: **incorrect top-1 count ; queries with no result ; mean minimum-score tie size**,
over eight queries. Critical/long cases are shown; all rows are in query_metrics.csv.
Dependency rejection is not counted as a correct returned match.

| Profile | Actions | v1 | v2 global | v2 scoped | v3 windows | v3 + dependencies | Exact |
|---|---|---|---|---|---|---|---|
| dense_target | 32 | 6;0;1.0 | 4;0;1.0 | 4;0;1.0 | 8;0;1.1 | 0;0;1.0 | 0;0;2.0 |
| dense_target | 1024 | 8;0;1.4 | 8;0;1.0 | 7;0;1.2 | 7;0;1.1 | 0;0;1.0 | 0;0;2.0 |
| dense_target | 4096 | 7;0;1.1 | 8;0;1.5 | 8;0;1.2 | 6;0;1.0 | 0;0;1.0 | 0;0;2.0 |
| sparse | 32768 | 5;0;1.5 | 0;0;2.0 | 0;0;2.0 | 0;0;2.0 | 0;0;2.0 | 0;0;2.0 |
| dense_unrelated | 32768 | 7;0;1.5 | 8;0;1.2 | 0;0;2.0 | 0;0;2.0 | 0;0;2.0 | 0;0;2.0 |
| dense_target | 32768 | 6;0;1.0 | 8;0;1.2 | 8;0;1.2 | 6;0;1.2 | 0;0;1.0 | 0;0;2.0 |
| dense_target_far | 4096 | 8;0;1.0 | 6;0;1.2 | 7;0;1.0 | 8;0;2.0 | 0;0;1.0 | 0;0;2.0 |
| dense_target_far | 32768 | 8;0;1.2 | 6;0;1.2 | 7;0;1.5 | 7;0;2.0 | 0;0;1.0 | 0;0;2.0 |

## Score separation

Mean irrelevant score minus relevant score, in Hamming-equivalent units. For v2 this
is literal 1,024-bit Hamming separation; for v3 it is the selected normalized window
cost. The longest histories are shown; all lengths remain in query_metrics.csv.
Positive is desirable; a distinct fingerprint alone does not prove retrieval.

| Profile | Actions | Representation | Separation |
|---|---|---|---|
| sparse | 32768 | v2_object_scoped | 154.32 |
| sparse | 32768 | w8_best | 154.32 |
| dense_unrelated | 32768 | v2_object_scoped | 154.32 |
| dense_unrelated | 32768 | w8_best | 154.32 |
| dense_target | 32768 | v2_object_scoped | 0.95 |
| dense_target | 32768 | w8_best | 16.29 |
| dense_target_far | 32768 | v2_object_scoped | -1.30 |
| dense_target_far | 32768 | w8_best | 3.68 |

## Resources and query latency

Index KiB is additional window records plus exact scoped references, per history.
GPU peak MiB is the explicitly allocated **whole-collection** query working set, not
per-history memory. Original v1/v2 memory remains separate in resources.csv.
Construction is the sum of CUDA-event window-kernel medians divided by history count.
Latency is **amortized batch query wall ms + selected CPU graph verification ms**, per
query. The GPU wall interval includes launches/results transfers and all three
aggregators/full-window top-K at the selected width, but excludes scalar validation.
It is not isolated single-query latency or full application end-to-end latency.
Raw CUDA medians and graph timings are retained separately. Display-GPU clocks vary.

| Profile | Actions | Anchors/windows | Added KiB | GPU peak MiB | Build ms | Query+graph ms |
|---|---|---|---|---|---|---|
| sparse | 32 | 5/3 | 0.6 | 0.032 | 0.002 | 0.037+0.009 |
| dense_unrelated | 32 | 5/3 | 0.6 | 0.032 | 0.002 | 0.037+0.010 |
| dense_target | 32 | 18/11 | 2.3 | 0.112 | 0.003 | 0.038+0.010 |
| sparse | 1024 | 5/3 | 0.6 | 0.032 | 0.002 | 0.038+0.011 |
| dense_unrelated | 1024 | 5/3 | 0.6 | 0.032 | 0.002 | 0.037+0.011 |
| dense_target | 1024 | 514/230 | 51.6 | 2.791 | 0.011 | 0.062+0.011 |
| sparse | 4096 | 5/3 | 0.6 | 0.032 | 0.002 | 0.038+0.012 |
| dense_unrelated | 4096 | 5/3 | 0.6 | 0.032 | 0.002 | 0.040+0.013 |
| dense_target | 4096 | 2050/902 | 203.1 | 11.088 | 0.038 | 0.132+0.019 |
| sparse | 32768 | 5/3 | 0.6 | 0.032 | 0.002 | 0.038+0.013 |
| dense_unrelated | 32768 | 5/3 | 0.6 | 0.032 | 0.002 | 0.042+0.013 |
| dense_target | 32768 | 16386/7174 | 1617.1 | 88.525 | 0.254 | 0.996+0.014 |
| dense_target_far | 4096 | 4095/1795 | 404.5 | 22.133 | 0.073 | 0.239+0.014 |
| dense_target_far | 32768 | 32767/14339 | 3232.5 | 177.008 | 0.544 | 2.234+0.015 |

At 32,768 actions, incremental representation/build costs compare as follows. The last
column is old four-method collection scoring CUDA time / new full-window plus three-
aggregator CUDA time, both amortized per query. These scopes differ and are labeled;
they are not interchangeable single-method latencies. V2 and v3 costs are additional
to their retained predecessors. Baseline timing repetition counts: [1];
v3 kernel repetition counts: [15].

| Profile | KiB v1/v2/v3 | Build ms v1/v2/v3 | Old/new CUDA query ms |
|---|---|---|---|
| sparse | 5301.7/1.3/0.6 | 0.561/0.447/0.002 | 0.007/0.006 |
| dense_unrelated | 5301.7/2816.6/0.6 | 0.622/1.071/0.002 | 0.006/0.005 |
| dense_target | 5301.7/2816.8/1617.1 | 0.742/1.467/0.254 | 0.004/0.574 |
| dense_target_far | 5301.7/5632.2/3232.5 | 0.740/2.820/0.544 | 0.005/1.055 |

## Correctness and limitations

For the held-out phase, 587,400 windows and 18,796,800 similarities
were checked. Every GPU window field/fingerprint, every full similarity-matrix value, parallel
window top-K, aggregate and history rank matches an independent scalar implementation.
Python independently checks archive metadata, sampled vector equations, scalar
Hamming/aggregation and ranks. Tests cover partial/long windows, identical multisets
in different orders, gap invariance, exact writer aliases, no label leakage and a
case with identical old answer keys but different raw source-version dependencies.

This study uses explicit object queries, boundary-context priors and a small known
grammar. It does not prove arbitrary interior-pattern discovery, unseen-object/grammar
generalization, full graph isomorphism, elapsed-time reasoning or policy enforcement.
Exact verification checks the endpoint-rooted dependency subgraph, not all observations.
The older exact baseline is comparison-only and never guides v3 retrieval.

Reproduce with scripts/run_temporal_v3.ps1 from a CUDA/MSVC developer shell. Binary
layouts, normalization, freeze rules and timing scope are documented in FORMAT.md.
No trained model/prototype is changed and no commit is created.
