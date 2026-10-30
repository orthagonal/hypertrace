# Temporal Memory v3 binary and retrieval contract

V3 is an additional index over the unchanged v2 object-scoped anchor sequence. The
original HDBTMP01 history, full v1 trees and HDBANC02 anchors remain authoritative.
No classifier or prototype is trained. All integer fields below are little-endian.

## Windows and exact references

For width W in {8,16,32}, stride is W/2. For a nonempty scoped sequence shorter than
W, store one partial window at zero. Otherwise store every stride-grid start less
than the scoped count, including partial trailing windows, and insert the complete
tail start `count-W` when not already on the grid. Starts are sorted and unique.
An empty sequence has no windows. Every anchor is covered; no history is truncated.

Window vectors use the v2 position binding, resetting ordinal to zero **inside each
window**. Per-bit votes are bundled once over the actual count, not padded to W:

```text
content  = anchor[i].hv[(word + (7*i)%32)%32]
bound    = rotate_left_32(content, i%32)
           xor mix32(0x71a5b39d xor i*0x9e3779b9 xor word*0x85ebca6b)
tie      = mix32(0xe17a1465 xor word*0xc2b2ae35)
```

Majority chooses the bit; equality chooses its deterministic tie bit. mix32 is the
unchanged v1/v2 equation. Original steps and writer references are retained exactly
but not hashed, so elapsed irrelevant gaps do not alter the local order encoding.

## Window archive: HDBWIN03

One file `windows_<history_id>.bin`, no labels/outcomes. Header is 64 bytes:

| Byte | Field | Value/meaning |
|---:|---|---|
| 0 | magic[8] | ASCII HDBWIN03 |
| 8 | version | 3 |
| 12 | header_bytes | 64 |
| 16 | history_id | Public metadata, not a feature |
| 20 | query_object | Explicit scoped object ID |
| 24 | scoped_anchor_count | N |
| 28 | total_window_count | Across all three widths |
| 32 | window_bytes | 176 |
| 36 | scoped_reference_bytes | 24 |
| 40 | reference_offset | 64 |
| 44 | descriptor_offset | 64 + 24N |
| 48 | width_count | 3 |
| 52 | descriptor_bytes | 24 |
| 56 | window_offset | descriptor_offset + 72 |
| 60 | reserved | 0 |

Each scoped reference is six u32 words:
`original_v2_anchor_ordinal, original_event_index, original_step, object_id,
object_writer_version, source_writer_version`.
Writer versions are exact observed writer steps, or `0xffffffff` when unavailable.
Source object IDs and complete provenance references remain in the referenced original
v2 anchor/Event; full dependency edges stay in the unchanged v1/v2 archives. References
to non-anchor producers remain valid. V3 does not substitute a summary for provenance.

Each width descriptor is six u32 words:
`width, count, stride, byte_offset, data_bytes, reserved`.

Each window is 176 bytes:

| Byte | Field |
|---:|---|
| 0 | hv[32]: 128 bytes |
| 128 | history_id |
| 132 | width |
| 136 | scope_begin (index in the scoped reference array) |
| 140 | actual_count |
| 144 | exact first_step |
| 148 | exact last_step |
| 152 | first original v2 anchor ordinal |
| 156 | last original v2 anchor ordinal |
| 160 | object ID |
| 164 | first object writer-version reference |
| 168 | last object writer-version reference |
| 172 | reserved, zero |

All intermediate anchor/version references are recoverable through the contiguous
scoped-reference range. Every window remains individually searchable. CUDA's
`compare_windows` accepts arbitrary stored query indices; `search_windows.py` is an
independent CPU archive-query/audit utility for any interior window.

## Complete similarity matrix: HDBSIM03

The 32-byte header contains magic[8] HDBSIM03 followed by six u32 words:
`version=3, header_bytes=32, width, query_rows, stored_windows, element_bytes=2`.
The body is row-major u16 Hamming counts, 0–1024. `queries.csv` maps each row to its
query history and global window ID. Global window IDs concatenate histories in ID
order at that width. Every stored window is compared, including partial windows.
A missing query context has score 1024. No HDC threshold excludes exhaustive results.
`window_topk.csv` contains parallel full-gallery window results, with ties by global ID.

## Count-normalized history scoring

Index storage and exhaustive window comparisons are separate from bounded history
aggregation. To give each history the same context budget, sample four query slots
and eight candidate slots at equally spaced ranks of its **complete** windows.
If none are complete, use all available partial windows. Rank for slot s is
`floor(s*(eligible_count-1)/(slots-1))`; small histories repeat their available context.
Thus every history gets eight scoring opportunities, not one opportunity per stored
window. This is an explicit query policy, not deletion of other windows. It can miss
arbitrary interior motifs; the complete index remains available for other queries.

For a four-by-eight matrix d of Hamming distances:

- **best**: the minimum of all 32 values.
- **normalized_top2**: for each query slot, take the mean of its two smallest candidate
  slot costs; average the four query costs. Repeated slots are retained in the budget.
- **ordered**: minimum mean cost of a strictly increasing four-slot alignment. Map
  the first query slot to candidate slot 0 and the last to slot 7; enumerate the two
  increasing middle slots. CPU computes the same result using dynamic programming.

Costs are integers scaled by 1024: best=min*1024, top2=sum_of_eight*128,
ordered=sum_of_four*256. No floating-point ranking/tie ambiguity is introduced.
History ranks sort (cost, public history ID), considering gallery roles only.
The equal opportunity cap, means, and extent constraint prevent raw window-count sums
or growing exhaustive minima from directly favoring longer histories.

## Optional exact dependency stage

Candidate proposals come from the chosen HDC history ranking. Only after proposal,
the CPU walks raw backward dependency edges from the first and last observed scoped
anchors in the query and candidate. It checks typed events, exact object/source IDs,
flags, edge roles, shared producer/version aliases, and relative chronology of mapped
events. Absolute step numbers need not match. Producer edges are canonicalized by
typed event fields; the final mapping must be injective and chronology-preserving.

This is an endpoint-rooted typed dependency-DAG verifier, **not** arbitrary graph
isomorphism or a comparison of the entire observed trace. It uses no v2 eight-word
answer key. A regression has equal old keys but different shared source versions and
must be rejected. Traversal is iterative and untruncated; a 32,768-node chain is tested.
No oracle label, future event or scenario ID chooses roots, proposals or graph matches.

The verification budget R is selected from {2,4,8,16} on validation. Among the first R
proposals, retain only exact matches, in approximate ranking order. The stage may
abstain. It cannot retrieve a correct history outside its proposed set. At R=16 it
checks the entire standard gallery, which must be reported as such. It produces no
allow/deny decision. `graph_checks.csv`, `graph_timings.csv` and `dependency_links.csv`
separate exact results/timing from HDC matching; link window ID `0xffffffff` means the
producer is outside the scoped window index and remains in the full history.

## Validation and held-out protocol

The original three profiles use 32/1024/4096/32768 actions. Four families, two orders
and three filler replicas give 16 gallery histories and eight held-out query histories
per collection. Relevance means same family **and order**, not the outcome alone.
Additional dense-target far cases separate critical operations by 256 or 2048 other
significant same-object operations. Exact edges connect them across distant windows.

Validation seed is 2027, test seed is 42. Before test generation/retrieval, freeze the
best mean validation window recall@2 (then hit@1; ties smaller width, then best/top2/
ordered). For that method choose the smallest graph budget achieving the best mean
validation dependency recall@2. No final-test result chooses a setting.

`frozen_settings.json` includes all validation ablations, source/data hashes and the
selection rule. Test generation and retrieval reject changed algorithm/metric hashes.
`evaluation_lock.json` records the exact frozen-file hash before test retrieval. The
test report uses that choice even if another test ablation happens to score higher.
The older exact chronological/provenance key is comparison-only; it never guides v3.

Recall@K divides by all relevant gallery histories (two by default); recall@1 therefore
has ceiling 0.5. Incorrect returned matches, abstentions, minimum-score tie sizes and
candidate recall are separate. Empty/missing contexts have maximum costs and ties,
not evidence of successful retrieval.

## CUDA timing, memory and independence

CUDA kernels construct windows, compare arbitrary query windows by XOR/popcount with
full-warp reductions, perform parallel stable top-K, aggregate costs and rank histories.
The target is sm_75. Each GPU window field/vector, matrix value and ranking is compared
against a scalar CPU implementation. CPU uses bit addressing/SWAR popcount and dynamic
programming; CUDA uses word rotation/popcount/shuffles and middle-slot enumeration.

Construction medians use CUDA events, two warmups and the requested repeat count.
Query batch wall time includes a single execution of all query kernels and result
transfers, but excludes independent scalar verification and repeated timing runs. Raw
kernel medians are also retained. Graph verification is CPU wall time, never presented
as CUDA time. Reports show amortized batch time per query, not isolated-query latency.

GPU memory counts explicit buffers for the whole collection at one width: scoped
anchors, window records, query/support metadata, full similarities and output ranks.
Widths are processed successively. Persistent archive memory counts all three widths
plus scoped references, additional to the complete v1/v2 representations. Driver,
allocator, host graph and application overhead are not included in device-byte totals.
