# Temporal Memory v1 measured report

This isolated experiment uses 8 full observed histories, 75,840 events,
and 36 separately stored exact dependency edges. Dataset SHA-256:
`10a870dd41bb99baf0aa8cdb4e33ed2c94b388d71171f30978b6933979d2a57f`.
The original v0 encoder, trained prototypes, CUDA learner and benchmarks are unchanged.
There is no new classifier. See FORMAT.md for every binary field and encoding equation.

## Correctness

All 75,840 GPU leaves and 2,448 internal nodes match the independent scalar
CPU implementation, including vectors, steps, child ranges and every mask bit.
Memory binaries and all deterministic CSVs match CPU/GPU byte for byte. Query distances
match independent XOR/bit counts; all eight exact property searches match linear scans.
Additional tests cover empty/singleton histories, 31/33/1,025 partial groups, histories
of 32,768 steps, future-record rejection, object IDs, outcome-metadata mutation, and
retrieval after summary corruption. Rust tests independently check deterministic pairs;
Python independently replays approval/version outcomes and checks the v0 byte hash.

## Resident tree memory by level (bytes)

Rows show one member of each pair. Leaves include exact events (32 bytes) and vectors
(128 bytes); internal records are 176 bytes, including eight exact 32-bit masks.

| History | Events | L0 | L1 | L2 | L3 | Tree total |
|---|---|---|---|---|---|---|
| 0 | 32 | 5,120 | 176 | 0 | 0 | 5,296 |
| 2 | 1024 | 163,840 | 5,632 | 176 | 0 | 169,648 |
| 4 | 4096 | 655,360 | 22,528 | 704 | 176 | 678,768 |
| 6 | 32768 | 5,242,880 | 180,224 | 5,632 | 176 | 5,428,912 |

Totals exclude input staging (32 bytes/event), query buffers, exact-search frontiers,
allocator overhead, and archived exact edges (12 bytes/edge). Edges are retained in
the dataset and memory file, not included in GPU HDC computations. Disk files also
have a 64-byte header and 24-byte descriptors per level. No full provenance graph
is claimed by masks.

## CUDA kernel measurements (milliseconds)

Hardware/implementation: Temporal Memory v1 CUDA | GPU: NVIDIA GeForce RTX 2060 SUPER | all nodes verified against independent CPU.
Measured with CUDA events, two warmups and repetition count(s)
[15]. Cells are medians. Summary-query and private-search columns are **sums
of separate kernel medians**, not end-to-end latency. Allocation, CPU verification,
transfers and file I/O are excluded. Other display workloads and clock variation can
affect these samples; this is a reproducibility measurement, not a tuned benchmark.

| Events | Encode | Build L1 | Build L2 | Build L3 | Query summaries | Private search |
|---|---|---|---|---|---|---|
| 32 | 0.006336 | 0.026624 | — | — | 0.005120 | 0.009856 |
| 1024 | 0.017440 | 0.027328 | 0.026624 | — | 0.009184 | 0.015232 |
| 4096 | 0.056064 | 0.027872 | 0.027136 | 0.009152 | 0.014560 | 0.019840 |
| 32768 | 0.383168 | 0.101632 | 0.027104 | 0.026624 | 0.015936 | 0.021856 |

The builder uses a warp per parent; query comparison uses XOR/popcount and full-warp
shuffle reduction. Exact navigation uses GPU frontier flags and property masks,
never HDC thresholds. Dense frontiers are a correctness baseline, not sparse traversal.

## Temporal order / removal / gap sensitivity

Distances below are root Hamming bit differences out of 1,024. Paired histories have
the same action/object/flag multiset, but opposite approval/rewrite order and opposite
later oracle outcomes. Controlled histories use a private read and approval among
identical noop fillers. Reorder swaps those two actions; removal drops the approval;
gap moves it from position 1 to position N−1. Steps remain exact but are not hashed.
These controls have no outcome labels or dependency edges and are not policy cases.

| Events | Paired order | Controlled order | Remove approval | Long gap |
|---|---|---|---|---|
| 32 | 70 | 83 | 105 | 79 |
| 1024 | 10 | 12 | 15 | 13 |
| 4096 | 4 | 5 | 8 | 8 |
| 32768 | 0 | 3 | 3 | 2 |

Local changes can be distinguishable at lower levels yet disappear after repeated
bundling. In this run, the policy order pair(s) at length(s) 32768 have root distance zero despite different policy outcomes. Thus a correct, order-sensitive implementation
does **not** imply a useful root representation. All aligned-level mean/max distances
are exported in sensitivity.csv. Removing an event can shift grouping boundaries;
moving it changes positional binding, not an independently modeled elapsed duration.
These few synthetic pairs do not establish general temporal reasoning or predictive
accuracy. Results for other seeds/pair counts must be measured, not assumed.

## Exact private-read retrieval

Both members of each pair have the same private-read locations. All eight property
queries are checked; this table illustrates the requested exhaustive private-read query.

| Events | Private reads found | Internal regions | False negatives | False positives |
|---|---|---|---|---|
| 32 | 8 | 1 | 0 | 0 |
| 1024 | 187 | 33 | 0 | 0 |
| 4096 | 773 | 132 | 0 | 0 |
| 32768 | 6524 | 1057 | 0 | 0 |

retrieved_events.csv lists every exact event location/object/flag, and regions.csv lists
every matching internal region and child mask. GPU vector words are deliberately poisoned
after similarity verification and before exact search; all exact results still match.
Exhaustive search stays exact even when summaries collide or are arbitrary. Property coincidence
does not prove that reads, approvals and transfers concern the same object/version;
the separately archived dependency edges provide those exact links.

## Reproduction and limits

Run scripts/run_temporal_v1.ps1 from a CUDA/MSVC developer shell. It builds the isolated
CPU/GPU programs, runs Rust/unit/integration tests, generates data and artifacts, and
rebuilds a report under target/temporal_v1. Individual commands are in this experiment's
README.md. Outcome metadata is separate from model input; the final transfer is never
exported. Oracle outcomes describe hypothetical later attempts, not an executed live
policy enforcer. No HTTP service, framework, classifier or commit is introduced.
