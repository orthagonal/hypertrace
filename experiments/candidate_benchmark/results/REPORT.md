# Temporal Memory v3.1: candidate retrieval benchmark

Measured on NVIDIA GeForce RTX 2060 SUPER, 610.88, 8192 MiB, 7.5. Validation seed 2027; held-out seed 42. No settings were selected on test results. The original frozen method remains `w8_best` (four query contexts, eight gallery contexts).

## Findings

Two gallery histories are relevant per query. Candidate recall is surviving relevant histories divided by two; final recall@2 counts relevant **returned** histories divided by two. Every K=1 result is capped at 50%. Rejected histories and abstentions contribute zero recall.

### How far can the HDC budget shrink?

Original v3 gallery, 32,768 events; candidate recall (HDC + verification has the same recall where verification is sound):

| Profile | K=1 | K=2 | K=4 | K=8 | K=16 | Smallest tested K retaining both matches |
|---|---:|---:|---:|---:|---:|---|
| sparse | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% | 2 |
| dense_unrelated | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% | 2 |
| dense_target | 12.5% | 12.5% | 31.2% | 75.0% | 100.0% | 16 |
| dense_target_far | 6.2% | 12.5% | 12.5% | 50.0% | 100.0% | 16 |

The dense failure remains a candidate-generation failure. Raising the verification budget cannot restore a relevant history that the shortlist excluded.

### Can exact indexing rank more reliably, and does HDC help after it?

Held-out 32,768-event histories, K=2 (candidate recall / final recall@2):

| Suite / G / profile | HDC + verify | Inverted + verify | Dependency join + verify | Dependency + HDC + verify | Inverted top-64 + HDC + verify |
|---|---:|---:|---:|---:|---:|
| original / 16 / sparse | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| original / 16 / dense_unrelated | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| original / 16 / dense_target | 12.5% / 12.5% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 12.5% / 12.5% |
| original / 16 / dense_target_far | 12.5% / 12.5% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 12.5% / 12.5% |
| scaled / 16 / dense_target | 68.8% / 68.8% | 56.2% / 56.2% | 100.0% / 100.0% | 100.0% / 100.0% | 68.8% / 68.8% |
| scaled / 128 / dense_target | 0.0% / 0.0% | 0.0% / 0.0% | 100.0% / 100.0% | 100.0% / 100.0% | 6.2% / 6.2% |
| scaled / 1024 / dense_target | 0.0% / 0.0% | 0.0% / 0.0% | 100.0% / 100.0% | 100.0% / 100.0% | 0.0% / 0.0% |

Dependency joins retain 100.0% of relevant candidates at K=2 across the held-out suite. The dependency/HDC hybrid retains 100.0%. Where the join already yields the two relevant histories, HDC has no ranking headroom. The additional inverted-top-64 hybrid tests an ambiguous shortlist; it is frozen at 64, with no test-driven threshold.

On the weaker inverted shortlist, HDC improves 24/56 paired test collections and hurts 16/56 at K=2. Mean final recall changes from 39.1% to 37.2%; occasional gains do not establish a reliable benefit.

### Scaling and cost

Scaled `dense_target`, 32,768 events. Single-query wall time includes scanning the complete observed query, window encoding, host/device transfers, ranking and verification. Resident indices are reused; original file loading and one-time CUDA context startup are excluded.

| Gallery | HDC build ms | Exact build ms | GPU similarity µs | HDC + verify K=16 ms | Dependency + verify K=2 ms | Hybrid K=2 ms | HDC candidate recall K=16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2.682 | 50.692 | 5.603 | 0.363 | 0.213 | 0.472 | 100.0% |
| 128 | 17.082 | 392.868 | 6.550 | 0.363 | 0.639 | 0.912 | 62.5% |
| 1024 | 131.081 | 3137.712 | 15.265 | 0.424 | 4.622 | 5.179 | 0.0% |

| Gallery | Exact shortlist µs | Verify shortlisted µs | Raw history MiB | Added exact index MiB (estimate) | GPU resident MiB | GPU batched pre-encoded queries/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 72.847 | 11.535 | 42.000 | 4.990 | 0.029 | 44277.175 |
| 128 | 496.908 | 13.410 | 238.000 | 37.667 | 0.142 | 36463.081 |
| 1024 | 4461.863 | 14.172 | 1806.000 | 306.147 | 1.052 | 23646.252 |

GPU similarity kernel time is not end-to-end latency. The batched figure amortizes transfers over eight already encoded queries and excludes query preparation and verification; it must not be compared directly with full single-query wall time. CSVs include kernel top-K time, complete query latency, graph work and CPU working-set measurements.

At matched 100% recall on the original long dense-target gallery, HDC K=16 takes 0.419 ms/query with 16 full verifications; the dependency index K=2 takes 0.185 ms/query with 2 verifications.

### GPU advantage and next direction

Assess retrieval cost at matched recall. On the original dense gallery, compare HDC K=16 with dependency K=2. On larger hard-negative galleries, an HDC budget that loses matches is not an equivalent faster solution. These results do not establish a useful end-to-end GPU retrieval advantage over the dependency index. GPU HDC stores a small support cache and provides fast similarity kernels, but those properties alone do not establish useful candidate reduction.

Pursue conventional object/version dependency indexing, with the unchanged full verifier retained for aliasing and chronology. Test its robustness on less synthetic dependency graphs before changing the HDC architecture. Do not start v4 based on fingerprint differences or kernel throughput alone.

## False matches, abstentions and work

At K=2, held-out totals across all completed original and scaled cases:

| Strategy | Mean candidate recall | Mean final recall@2 | False returned histories | Zero-result queries | Full verifications |
|---|---:|---:|---:|---:|---:|
| random_only | 4.8% | 4.8% | 853 | 0 | 0 |
| hdc_only | 32.1% | 32.1% | 608 | 0 | 0 |
| inverted_only | 39.1% | 39.1% | 546 | 0 | 0 |
| dependency_only | 100.0% | 100.0% | 0 | 0 | 0 |
| random_verified | 4.8% | 4.8% | 0 | 405 | 896 |
| hdc_verified | 32.1% | 32.1% | 0 | 264 | 896 |
| inverted_verified | 39.1% | 39.1% | 0 | 238 | 896 |
| dependency_verified | 100.0% | 100.0% | 0 | 0 | 896 |
| hybrid_verified | 100.0% | 100.0% | 0 | 0 | 896 |
| hybrid_inverted_verified | 37.2% | 37.2% | 0 | 236 | 896 |

## Verification and limitations

All 112 collections passed independent scalar CPU/GPU fingerprint, similarity-score and top-16 ranking comparisons, including masked hybrid searches. Original archived `w8_best` scores are compared exactly for all original collections. Prior source/test hashes are unchanged. Regression coverage includes empty/partial windows, 1,024-way ties, 32,768-step histories, future-event rejection, metadata isolation and a source-version alias that survives the semijoin but fails full verification.

The original G=16 benchmark is preserved. Larger galleries use a separate nested suite, with a scaled G=16 control: 256 distinct ordered writer chains made from the same action multiset, identical endpoint types and shared object IDs. Two independently filled replicas are relevant; opposite-order replicas are hard negatives. Results from the scaled suite must not be attributed solely to gallery size by comparing them directly with the original grammar.

Only four query families (eight queries) are used per collection, and the two phase seeds are not confidence intervals. Synthetic Filler flags and clean writer edges make the conventional join easier than uncertain real provenance. The join is a necessary-condition filter, not full graph isomorphism; it may admit alias/chronology errors on richer data. The verifier only compares its existing endpoint-rooted dependency DAG, not arbitrary whole-history policy equivalence. Raw histories are kept in full; only frozen HDC support contexts are cached. Neither HDC nor retrieval recall is exact policy enforcement.

Construction excludes validation/reference checks. GPU memory is allocated buffer payload, not driver/context overhead; CPU exact-index bytes are an allocation estimate, with sampled process working set separately reported. Existing verifier node counts record successfully mapped nodes before failure, a lower bound on all comparisons; semijoin node/posting/edge work is counted separately. GPU top-K always emits the same 16-rank prefix, so K varies verification budget rather than the ranking algorithm. Latencies are arithmetic means of the configured repeated runs on this machine, not service-level guarantees.

Full profile × length × gallery × budget results, both validation and test: [summary.csv](summary.csv). Construction/memory/batched throughput: [construction.csv](construction.csv). Hamming separation: [separation.csv](separation.csv). Source/settings/output hashes and hardware: [provenance.json](provenance.json). Per-query outcomes and candidate acceptance traces remain in the reproducible runner output under `target/candidate_benchmark`.
