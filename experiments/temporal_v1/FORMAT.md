# Temporal Memory v1 binary contract

All integers are unsigned **little-endian u32**, with no alignment padding on disk.
Offsets are byte offsets from the beginning of a file unless stated otherwise.
Dimensionality is fixed at 1,024 bits (32 words), fanout at 32, and property count at 8.
Reserved fields must be zero. Inputs larger than two million events are rejected by
the experiment reader; this is an allocation bound, not a 1,024-step limit.

## Observed dataset: `HDBTMP01`, version 1

This is a new format and grammar. It does not reinterpret or replace `HDBSAFE1` v0.
The optional Rust switch `--temporal-v1` selects it; normal invocation is unchanged.

### Header, 64 bytes

| Byte | Field | Value / meaning |
|---:|---|---|
| 0 | magic[8] | ASCII `HDBTMP01` |
| 8 | version | 1 |
| 12 | header_bytes | 64 |
| 16 | history_count | H |
| 20 | event_count | E, across all histories |
| 24 | edge_count | D, across all histories |
| 28 | event_bytes | 32 |
| 32 | edge_bytes | 12 |
| 36 | descriptor_bytes | 32 |
| 40 | descriptor_offset | 64 |
| 44 | event_offset | 64 + 32H |
| 48 | edge_offset | event_offset + 32E |
| 52 | outcome_offset | edge_offset + 12D |
| 56 | outcome_bytes | 16 |
| 60 | reserved | 0 |

Sections are contiguous; trailing bytes, truncated files, bad offsets, unsupported
versions and malformed dependency ranges are rejected.

### History descriptor, 32 bytes

Eight words: `history_id, pair_id, event_begin, event_count, edge_begin, edge_count,
observed_cutoff_step, reserved`. IDs are sequential, zero based. Begin fields are
record indices in their respective global tables. Steps must be strictly increasing
within each history and less than its cutoff. Empty histories have cutoff zero.

### Exact event, 32 bytes

| Byte | Field | Meaning |
|---:|---|---|
| 0 | action | Exact action code |
| 4 | object | Target/result object ID; 0 means none |
| 8 | source | Source object ID; 0 means none |
| 12 | flags | Exact observed event properties below |
| 16 | step | Original observed sequence step |
| 20 | dependency_begin | First edge index **local to this history's edge table** |
| 24 | dependency_count | Number of exact dependency records owned by this event |
| 28 | reserved | 0 |

Property bits, in mask-array order:

| Bit | Property | Semantics |
|---:|---|---|
| 0 | private_read | Includes unrelated private reads |
| 1 | external | Observed web search or external staging; not a future POST |
| 2 | approval | Approval event for the target object's current version |
| 3 | derived | Derived/re-written object |
| 4 | sensitive | Observed content is sensitive |
| 5 | local | Local edit operation |
| 6 | untrusted | Observed untrusted external destination |
| 7 | filler | Background action on an unrelated object |

Action codes retain the existing meanings: 1 private read, 4 transform/rewrite,
6 local search, 7 web search, 8 unrelated private read, 11 approve, 13 stage external,
15 unrelated edit, 17 noop. No final HTTP transfer is exported. Payload source is
object 1, working payload is object 2, and filler objects are 1000–1031. IDs are exact,
not hashes; only their HDC representations are lossy.

### Exact dependency edge, 12 bytes

Three words: `consumer_step, producer_step, object_id`. All references stay inside
the observed history and point backwards to an existing event on that object.
Derivation edges identify the source version's writer. Approval edges identify the
approved version's writer. Staging retains the current writer and valid approval
event, when present. Rewriting invalidates previous approval and creates a new version
identified by its writer step. The reader checks ownership, object IDs and backward
references. These edges are retained separately and are **not** reconstructed from
property masks or HDC summaries. Filler objects never modify payload objects 1/2.

### Outcome table, 16 bytes per history

Four words: `history_id, oracle_label, scenario_type, violation_category`.
Label 1 means a hypothetical later unauthorized sensitive external transfer; 0 means
allowed. Scenario 1 is approval/rewrite ordering (0 for histories too short for the
scenario). Category 1 is sensitive external transfer without current-version approval;
0 means no violation. This table is separate from `History`, is never uploaded to
CUDA, and is not serialized in temporal memory files.

Each pair has identical action/object/flag multisets and filler draws. The safe order
rewrites then approves; the unsafe order approves then rewrites. The observed prefix
ends at staging. The oracle evaluates a hypothetical later transfer separately.
Default lengths are 32, 1,024, 4,096 and 32,768. Lengths below 5 are format edge cases,
not order-sensitive policy scenarios. Seed, requested length order and pair count
determine histories; v1 uses a separate RNG stream from v0.

## Built hierarchy: `HDBMEM01`, version 1

One file per history, containing exact events, their HDC vectors, exact dependency
edges, and every parent. It contains no oracle label, scenario or violation category.

### Header, 64 bytes

| Byte | Field | Value / meaning |
|---:|---|---|
| 0 | magic[8] | ASCII `HDBMEM01` |
| 8 | version | 1 |
| 12 | header_bytes | 64 |
| 16 | history_id | Metadata only, never encoded |
| 20 | event_count | N |
| 24 | edge_count | D |
| 28 | level_count | Includes level 0, even when empty |
| 32 | leaf_bytes | 160 |
| 36 | node_bytes | 176 |
| 40 | edge_bytes | 12 |
| 44 | level_descriptor_bytes | 24 |
| 48 | level_table_offset | 64 |
| 52 | leaf_offset | 64 + 24 * level_count |
| 56 | edge_offset | leaf_offset + 160N |
| 60 | parent_offset | edge_offset + 12D |

Each level descriptor is six words: `level, count, record_bytes, data_offset,
data_bytes, reserved`. Edges follow level 0; parent levels follow edges in increasing
level order. A singleton has only a leaf root. Empty input has no root and no CUDA
launches, with one zero-count level descriptor. No padding children are invented.

### Level 0: 160-byte leaf

The exact 32-byte event above, followed by `hv[32]` (128 bytes). Dependency indices
address this memory file's exact edge section. Edges are archived on the host and
are not used by the GPU HDC builder; no GPU provenance-graph query is claimed.

### Levels 1+: 176-byte node

| Byte | Field |
|---:|---|
| 0 | hv[32], 128-byte HDC summary |
| 128 | start_step, the earliest descendant's original step |
| 132 | child_index, record index in the immediately lower level |
| 136 | child_count, 1–32 real children |
| 140 | properties, exact OR of all descendant event property flags |
| 144 | masks[8], 32 bytes, in the property order above |

Bit j of masks[p] is 1 iff child j's subtree contains property p. Bits beyond
child_count are zero. Masks say nothing about whether a private read, approval and
external operation refer to the same object or version. Use the exact edge table
and raw events for those relationships. A parent mask is not a provenance graph.

## Encoding and deterministic ties

All arithmetic below wraps modulo 2^32. Define:

```text
hash(x): x=(x xor (x>>16))*0x7feb352d
         x=(x xor (x>>15))*0x846ca68b
         return x xor (x>>16)
```

An event has eleven tokens: action (tag 1), object (2), source (3), and the eight
Boolean property states (tags 4–11, including false states). Token word w is
`hash(tag*0x9e3779b9 xor value xor w*0x85ebca6b xor 0xa5f152bd)`.
Bundle by per-bit majority over eleven tokens (no ties). Step and dependency offsets
are retained exactly but are **not** hashed: chronology comes from sequence position.
No absolute time-gap model, label, scenario ID or outcome-dependent field is encoded.

For parent level L, local child position c (0–31), and output word w:

```text
content = child.hv[(w + (7*c + L) % 32) % 32]
bound   = rotate_left_32(content, c)
          xor hash(0x51ed270b xor L*0x9e3779b9 xor c*0x85ebca6b xor w*0xc2b2ae35)
tie     = hash(0xd1b54a35 xor L*0x94d049bb xor w*0x369dea0f)
```

Bundle each bound bit over the actual child_count. More than half ones gives 1;
less than half gives 0; equality uses the corresponding tie bit. The position
transform includes a word permutation, bit rotation and positional XOR. No random
runtime state is used. Successive levels repeat this process until one root remains.
The scalar CPU implementation uses explicit bit addressing; CUDA uses word rotation.

## Exact queries versus approximate similarity

Similarity compares every requested leaf/summary with a query using XOR+popcount,
then a full-warp shuffle reduction. The default query is the earliest private-read
event's vector, or a fixed private-read vector if no such observation exists.

Exhaustive property search starts at the root, descends **only exact child masks**,
and verifies the property on leaves. CUDA produces frontier flags, exact event indices
and matching internal regions; the CPU reference and a linear scan check all results.
Neither search implementation reads HDC vectors, similarity thresholds or outcomes.
As a correctness challenge, after query-distance verification the GPU test path overwrites
all device vector words before exact search. Host archives retain the verified original
vectors, and every retrieved event/region must still match the independent reference.
Navigation uses dense GPU frontiers, not a sparse/compacted work queue; performance
optimization is outside this first correctness experiment.
