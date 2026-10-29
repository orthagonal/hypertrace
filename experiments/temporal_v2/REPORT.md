# Temporal Memory v2: Sparse Anchors — measured report

Temporal Memory v2 CUDA | GPU: NVIDIA GeForce RTX 2060 SUPER. CUDA repetition counts: [15]; two warmups per kernel.
The original v0/v1 sources, full tree, tests and trained prototypes are unchanged.
Every selected anchor, exact writer version, event vector, fingerprint, all-pairs score
and GPU top-K result matches the independent scalar CPU reference. Full v1 trees are
kept in separate GPU buffers and checked unchanged after sparse construction.

- sparse, 32,768 actions: recall@2 is 37.5% / 100.0% / 100.0% / 100.0% (v1 / global / scoped / exact).
- dense_unrelated, 32,768 actions: recall@2 is 12.5% / 0.0% / 100.0% / 100.0% (v1 / global / scoped / exact).
- dense_target, 32,768 actions: recall@2 is 25.0% / 0.0% / 0.0% / 100.0% (v1 / global / scoped / exact).

## Retrieval task

Each collection has 24 independently generated histories. 4
motif families each have two approval/rewrite orders with opposite later outcomes,
2 gallery replicas and one held-out query replica (16 gallery, 8 queries). Object
IDs 1/2 are shared by all families: object identity alone cannot solve retrieval.
Paired orders have exactly the same action/object/flag multiset. Replica filler draws
are independent. Relevance requires the same **family and order**, not just the same
violation label. Oracle labels/catalog categories are used only by this report scorer.

The exact baseline matches an explicitly defined observed chronological/provenance
signature: first/latest target write, source, last approval's relation to the current
writer version, and staging fields. It is not a claim to match arbitrary provenance
graphs. No classifier or policy enforcement decision is produced.

`v2_global` flat-bundles every selected anchor with its ordinal; `v2_object_scoped`
does the same for explicit query object 2. Scoping is a query constraint, not a learned
discovery of relevant objects. All background anchors remain in the index. The scoped
encoding is not assigned extra bits. Steps/versions remain exact in anchors but are
not hashed, so irrelevant gaps do not perturb anchor ordinal positions.

## Recall as history length increases

Cells are **recall@1 / recall@2 (%)**. Each query has 2 relevant gallery histories,
so perfect top-1 retrieval gives recall@1=50.0%.
All methods use the same candidates and deterministic (distance, history-ID) ordering;
IDs were shuffled independently of outcome. Full recall@5, wrong matches, zero-distance
false matches and tie-aware top-1 success are in query_metrics.csv.

### sparse

| Actions | v1 root | v2 global | v2 scoped | Exact pattern |
|---|---|---|---|---|
| 32 | 25.0 / 31.2 | 50.0 / 100.0 | 50.0 / 100.0 | 50.0 / 100.0 |
| 1024 | 18.8 / 31.2 | 50.0 / 100.0 | 50.0 / 100.0 | 50.0 / 100.0 |
| 4096 | 12.5 / 18.8 | 50.0 / 100.0 | 50.0 / 100.0 | 50.0 / 100.0 |
| 32768 | 18.8 / 37.5 | 50.0 / 100.0 | 50.0 / 100.0 | 50.0 / 100.0 |
### dense_unrelated

| Actions | v1 root | v2 global | v2 scoped | Exact pattern |
|---|---|---|---|---|
| 32 | 0.0 / 6.2 | 25.0 / 31.2 | 50.0 / 100.0 | 50.0 / 100.0 |
| 1024 | 6.2 / 12.5 | 6.2 / 6.2 | 50.0 / 100.0 | 50.0 / 100.0 |
| 4096 | 12.5 / 25.0 | 12.5 / 25.0 | 50.0 / 100.0 | 50.0 / 100.0 |
| 32768 | 6.2 / 12.5 | 0.0 / 0.0 | 50.0 / 100.0 | 50.0 / 100.0 |
### dense_target

| Actions | v1 root | v2 global | v2 scoped | Exact pattern |
|---|---|---|---|---|
| 32 | 12.5 / 25.0 | 25.0 / 31.2 | 25.0 / 31.2 | 50.0 / 100.0 |
| 1024 | 0.0 / 0.0 | 0.0 / 0.0 | 6.2 / 12.5 | 50.0 / 100.0 |
| 4096 | 6.2 / 6.2 | 0.0 / 18.8 | 0.0 / 0.0 | 50.0 / 100.0 |
| 32768 | 12.5 / 25.0 | 0.0 / 0.0 | 0.0 / 0.0 | 50.0 / 100.0 |

## False top-1 matches at the longest tested length

Counts are wrong temporal patterns / queries. Fixed top-K always returns K available
candidates, even for the exact baseline: at K=5, at least 3 are irrelevant because
only 2 relevant candidates exist. Interpret false@5 with that constraint.

| Profile | v1 | v2 global | v2 scoped | Exact |
|---|---|---|---|---|
| sparse | 5/8 | 0/8 | 0/8 | 0/8 |
| dense_unrelated | 7/8 | 8/8 | 0/8 | 0/8 |
| dense_target | 6/8 | 8/8 | 8/8 | 0/8 |

## Opposite-order pair Hamming distances

Mean [min,max], out of 1,024 bits, over all matched replica/family pairs. These distances
are diagnostics; the recall tables above determine whether they support retrieval.

| Profile | Actions | v1 root | v2 global | v2 scoped |
|---|---|---|---|---|
| sparse | 32 | 75.1 [62,92] | 167.8 [163,174] | 198.2 [190,207] |
| dense_unrelated | 32 | 75.7 [66,85] | 108.5 [92,120] | 198.2 [190,207] |
| dense_target | 32 | 78.2 [69,98] | 104.2 [91,113] | 110.2 [97,132] |
| sparse | 1024 | 15.3 [12,18] | 167.8 [163,174] | 198.2 [190,207] |
| dense_unrelated | 1024 | 12.2 [7,19] | 18.9 [8,30] | 198.2 [190,207] |
| dense_target | 1024 | 10.2 [6,14] | 21.3 [13,27] | 21.1 [10,29] |
| sparse | 4096 | 5.3 [2,9] | 167.8 [163,174] | 198.2 [190,207] |
| dense_unrelated | 4096 | 4.9 [3,8] | 11.2 [7,18] | 198.2 [190,207] |
| dense_target | 4096 | 3.7 [2,5] | 10.1 [6,15] | 9.4 [5,16] |
| sparse | 32768 | 1.6 [0,6] | 167.8 [163,174] | 198.2 [190,207] |
| dense_unrelated | 32768 | 1.9 [0,4] | 4.1 [1,8] | 198.2 [190,207] |
| dense_target | 32768 | 1.7 [0,3] | 3.1 [0,6] | 4.5 [2,7] |

## Memory and CUDA execution time

Means per history. `anchors` is all/scoped count. Memory is KiB; the v2 index is
**additional** to v1 and includes full 176-byte anchor records plus two fingerprints.
Exact edges are archived separately (12 bytes each). Rows do not include scratch,
input staging, the 32-byte exact-baseline key or allocator overhead; resources.csv
reports a conservative scratch upper bound. No anchors are truncated in dense cases.

Timing cells are **v1 / v2 build milliseconds**, each a sum of separate kernel medians,
averaged across histories, not end-to-end latency. V2 includes selection, stable GPU
scan/compaction, writer-version reconstruction, anchor encoding, scope selection and
flat bundling. It excludes host object-ID enumeration, transfers, allocation, file I/O
and CPU verification. GPU collection-scoring/top-K and exact-key timings are separate
in resources.csv and raw timings.csv. This is an untuned experiment on a display GPU.

| Profile | Actions | Anchors all/scoped | v1 KiB | Added v2 KiB | Build ms v1/v2 |
|---|---|---|---|---|---|
| sparse | 32 | 6/5 | 5.2 | 1.3 | 0.033/0.047 |
| dense_unrelated | 32 | 18/5 | 5.2 | 3.3 | 0.033/0.048 |
| dense_target | 32 | 19/18 | 5.2 | 3.5 | 0.033/0.051 |
| sparse | 1024 | 6/5 | 165.7 | 1.3 | 0.071/0.066 |
| dense_unrelated | 1024 | 513/5 | 165.7 | 88.4 | 0.071/0.103 |
| dense_target | 1024 | 515/514 | 165.7 | 88.8 | 0.071/0.156 |
| sparse | 4096 | 6/5 | 662.9 | 1.3 | 0.120/0.098 |
| dense_unrelated | 4096 | 2049/5 | 662.9 | 352.4 | 0.120/0.182 |
| dense_target | 4096 | 2051/2050 | 662.9 | 352.8 | 0.120/0.229 |
| sparse | 32768 | 6/5 | 5301.7 | 1.3 | 0.426/0.286 |
| dense_unrelated | 32768 | 16386/5 | 5301.7 | 2816.6 | 0.421/0.661 |
| dense_target | 32768 | 16387/16386 | 5301.7 | 2816.8 | 0.424/0.722 |

## Interpretation and limits

The measured sparse case preserves the requested patterns across long irrelevant gaps.
Object scoping also succeeds when many significant events concern unrelated objects.
However, the measured dense-target case remains unsolved by v2: flat majority bundling
still dilutes differences when many anchors concern the same queried object. This is
an experimental limitation, not a correctness failure. Distinct fingerprints alone
are insufficient; the held-out retrieval results expose that distinction.

This corpus is a controlled, single-seed, 4-family retrieval task with explicit
object queries. It does not establish unseen-object, unseen-grammar, arbitrary graph,
elapsed-time or security-enforcement generalization. Full provenance references remain
available through the original v1 events/edges, including non-anchor producers. Writer
steps are observed version identifiers; no hidden version counter is invented.

## Reproduction

Run `scripts/run_temporal_v2.ps1` from a CUDA/MSVC developer shell. The default workflow
builds, runs all existing v0/v1 regressions and new sparse tests, generates the corpus,
checks every CPU/GPU artifact, and recreates this report. See README.md and FORMAT.md
for exact significance rules, binary layout, equations and metric definitions.
No classifier, prototype change or automatic commit is introduced.
