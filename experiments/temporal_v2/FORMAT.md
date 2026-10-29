# Sparse Anchors v2 contract

V2 is an additional index over **unchanged** v1 `HDBTMP01` histories. It does not
replace the full event tree, reinterpret v0 records, train prototypes or create a
classifier. All integers below are little-endian u32. See ../temporal_v1/FORMAT.md
for exact event, edge and full-tree layouts.

## Anchor significance

An event is selected if **any** of these observed conditions holds:

| Condition | Exact rule |
|---|---|
| Sensitive read | Action 1/2/8 or `PrivateRead`, **and** `Sensitive` |
| Transformation/copy | Action 3/4/16 or `Derived` |
| Save/stage | Action 5/13/14 |
| Approval/request | Action 11/12 or `Approval` |
| External operation | `External` |

These are observed action/property tests. The `Filler` flag is **not** an exclusion:
private filler reads, unrelated derivations, approvals and external operations remain
anchors. Action 12 is an approval request, not a fabricated grant. The v1 vocabulary
has no explicit permission-revocation event; this experiment does not invent one.
No event is selected from its future outcome, scenario, pair ID or oracle label.

Selection and compaction are stable. A GPU integer exclusive scan assigns ordinals
in original chronological order. No anchor is truncated, including dense streams.
Steps may have gaps; chronological order is the input order, validated by the v1 reader.

## Exact anchor and provenance

Each anchor is 176 bytes:

| Offset | Field | Bytes |
|---:|---|---:|
| 0 | Original v1 Event: action, object, source, flags, step, dependency_begin, dependency_count, reserved | 32 |
| 32 | event_index in the original full history | 4 |
| 36 | object_version | 4 |
| 40 | source_version | 4 |
| 44 | ordinal in the full anchor sequence | 4 |
| 48 | event HDC vector, 32 words | 128 |

Version IDs are **observed writer steps**, not invented counters. Writers are actions
1/2/3/4/8/15/16. object_version is the latest observed writer through the event itself;
source_version is the latest source writer strictly before the event. `0xffffffff`
means no writer is available. Absolute writer steps are exact metadata, not HDC tokens.
An approval therefore retains the writer version it concerned; a later rewrite has
a different writer step. These fields do not assert that approval remains valid forever.

All original dependency edges are copied intact to the sparse archive. The anchor's
dependency_begin/count remain indices in that history's **full edge table**. Edges
may reference non-anchor events: use event_index/step and the unchanged full history
to resolve them. Dropping a nonsignificant event from the index does not delete it
or its provenance from v1. No approximate vector stands in for a provenance graph.

CPU version reconstruction uses a chronological object-to-writer map. CUDA independently
uses per-object prefix maxima of writer indices, resolving sources before writes and
targets after writes. The host enumerates distinct observed object IDs only to schedule
GPU blocks; version computation and anchor selection are performed on the GPU.

## Archive: `HDBANC02`, version 2

One file `anchors_<history_id>.bin` per history. No label, outcome, scenario or role
is serialized. Header is 64 bytes:

| Byte | Field | Meaning |
|---:|---|---|
| 0 | magic[8] | ASCII `HDBANC02` |
| 8 | version | 2 |
| 12 | header_bytes | 64 |
| 16 | history_id | Metadata, never encoded |
| 20 | event_count | Full observed length |
| 24 | anchor_count | M |
| 28 | query_object | Explicit observed object ID supplied by the query |
| 32 | scoped_count | Anchors with object == query_object |
| 36 | anchor_bytes | 176 |
| 40 | edge_count | D |
| 44 | edge_bytes | 12 |
| 48 | anchor_offset | 64 |
| 52 | edge_offset | 64 + 176M |
| 56 | tail_offset | edge_offset + 12D |
| 60 | tail_words | 104 |

Tail consists of three 1,024-bit vectors (v1 root, v2 global, v2 object-scoped), then
the eight-word exact pattern key below. The v1 root is archived for comparison; its
128 bytes are not counted as additional v2 fingerprint memory. The complete v1 tree
is built with its unchanged implementation and remains in disjoint GPU buffers during
sparse construction. Every byte of those buffers is checked unchanged afterward.
`v1_history_0.mem` archives one full tree per collection; the source dataset retains
every event and edge for all histories. No full tree is replaced by the sparse index.

## Deterministic 1,024-bit positional encoding

Anchor event vectors use the unchanged v1 eleven-token event encoding: action,
object, source and eight property states. CPU and CUDA implement it independently.

The global fingerprint flat-bundles all anchor vectors, without recursive majority
levels. The scoped fingerprint selects anchors whose **target object** equals the
explicit query object, retaining their order and assigning scope-local ordinals.
It does not automatically discover relevant objects or compute a provenance closure.
Source IDs and original provenance references are retained in the selected anchors.

For ordinal i and output word w (all arithmetic wraps modulo 2^32):

```text
content  = anchor[i].hv[(w + (7*i)%32)%32]
position = hash(0x71a5b39d xor i*0x9e3779b9 xor w*0x85ebca6b)
bound    = rotate_left_32(content, i%32) xor position
tie      = hash(0xe17a1465 xor w*0xc2b2ae35)
```

`hash` is the v1 mix32 equation. For each bit, majority over all retained bound bits
chooses 0/1; equality chooses the corresponding tie bit. The denominator is the real
anchor count, never padded. Empty input produces the all-zero sentinel. Global and
scoped fingerprints receive the same 1,024-bit budget and tie rule.

Absolute step, gap, writer version, dependency offsets, history ID, label and outcome
are not hashed. Therefore inserting irrelevant operations leaves both sparse order
fingerprints unchanged, while exact original steps remain available for precise gap
queries. This is order indexing, not elapsed-time encoding. Many significant events
can still dilute a flat majority; measurement is required.

CUDA counts bound votes in 256-anchor tiles, then sums integer votes once. This final
sum is exact and does not majority-bundle intermediate summaries. Stable compaction
uses a recursive 256-wide GPU scan; its offset-add kernel overwrites output, making
all timing repetitions idempotent.

## Exact observed pattern baseline

The retrieval query requests object 2 and a finite chronological/provenance pattern.
The exact baseline key is eight words:

1. First target writer's action (0 if absent).
2. Latest target writer's action (0 if absent).
3. Whether the latest approval follows the latest writer: 0/1, or 2 if unavailable.
4. Latest observed staging action 13/14 (0 if absent).
5. Explicit target object ID.
6. Latest target writer's source object ID.
7. Latest staging flags masked to Sensitive/Untrusted/External.
8. Whether that approval's referenced writer version equals the current writer.

This signature is deliberately defined for the controlled retrieval task. It is not
an arbitrary-graph matcher and does not output a policy allow/deny decision. CPU and
GPU derive it separately from observed events and writer versions. Equality scores
0, inequality 1024. No oracle label is involved.

## Gallery/query separation, scoring and false matches

The standard corpus has four motif families (first/latest writer action pairs 4/4,
3/16, 16/3 and 4/16), two approval/rewrite orders, and three replicas. Two replicas
form a 16-history gallery; the third forms eight held-out query histories. Every
query has exactly two relevant gallery histories: same family **and order**.
Neither binary outcome alone nor object identity alone defines relevance.

`roles.csv` contains only `history_id,role`, with 0 gallery / 1 query. It controls
candidate eligibility, not fingerprint encoding. Scoring-only `catalog.csv` contains
family/order/replica/oracle metadata and is never opened by CPU/CUDA executables.
History IDs are deterministically shuffled. Rankings use increasing distance, then
increasing public history ID; no label-dependent tie-break is permitted.

CUDA computes all pairwise Hamming distances with XOR/popcount/warp reductions and
top-5 gallery rankings. Every score and rank is compared with independent CPU code.
Python independently recomputes Hamming distances/rankings and evaluates relevance.

- Recall@K = relevant retrieved in top K / all relevant gallery histories.
- Hit@1 = whether the first candidate is relevant. With two relevant candidates,
  recall@1 has ceiling 0.5 even when Hit@1=1.
- False@K = count of irrelevant returned candidates. All methods return up to K
  available candidates; at K=5, three false matches are unavoidable with two relevant.
- Zero-distance false matches = irrelevant candidates with score zero, before top-K.
- Tie-aware Hit@1 = fraction of relevant candidates among all minimum-distance ties;
  this audits lucky/unlucky deterministic ties without changing the returned ranking.

## Dense controls and memory/timing accounting

Sparse filler contains local searches, noops and unrelated nonsensitive edits. In
dense_unrelated, roughly half the operations are genuinely significant actions on
other objects; they remain anchors. In dense_target, roughly half concern the queried
object itself. The final order-dependent rewrite/approval pair follows that background,
so the two later outcomes remain controlled. Paired orders share identical filler;
gallery/query replicas use independent filler draws.

Memory reports count full anchor records plus two v2 fingerprints, additional to
the complete v1 tree. They separately expose exact-key/edge storage and a conservative
scratch upper bound (input, scans, version map/object IDs, vote tiles, output buffers).
CUDA timings use events, two warmups and repeated medians. Sums of per-kernel medians
are not end-to-end latency. Host object enumeration, allocation/copies, file I/O and
independent CPU verification are excluded and are not disguised as GPU execution.
