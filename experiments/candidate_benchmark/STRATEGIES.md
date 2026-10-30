# Candidate-generation strategies and measurement contract

## Inputs and relevance

All strategies receive the same observed `History` objects and gallery/query
roles. They never receive the separate `Truth` family/order/replica table,
oracle outcomes, pair IDs, scenario IDs, future actions or precomputed answer
signatures. Family/order is consulted only by the evaluator after ranking.
Object IDs intentionally overlap between families. Writer step numbers are
local to a history: joins follow their actual edges instead of comparing step
IDs across histories. Events must precede the observation cutoff, and writers
must precede consumers.

Relevance is the existing same-family-and-order definition. There are exactly
two relevant gallery histories per query, including on larger galleries.
Candidate recall@K is relevant histories in the first K candidates divided by
two. Final recall@2 is relevant histories in the final, at-most-two returned
results divided by two. K=1 cannot exceed 50%. False matches count returned
irrelevant histories. An abstention means no returned result; `partial` also
counts queries returning fewer than two. Rejections and abstentions add no hits.

## Ranking and verification

All rankings break ties by ascending public gallery history ID. IDs are shuffled
independently of outcomes; query IDs are shuffled as well. K is 1, 2, 4, 8 or 16.
The GPU always computes the same top-16 prefix. K changes only how many are
considered or verified. No query returns a candidate outside that budget.

* **Random:** fixed Fisher-Yates ranking, seeded by 173 plus the public query ID;
  no observed or evaluation metadata influences its draws.
* **HDC:** exactly the frozen v3 `w8_best` score: minimum Hamming distance over
  four query windows and eight candidate windows of eight object-scoped anchors.
  Scores are distances multiplied by 1,024, as in v3. The full original window
  selection, full-tail rule, event encoder, positional binding and tie-breaking
  are preserved. Only these participating contexts need caching: other v3
  windows cannot affect this frozen history score. The cache is additional to
  the unchanged full v1/v2/v3 representations. It is checked against archived
  original v3 scores and an independent scalar reference.
* **Inverted index (A):** postings of histories containing primitive object IDs,
  event types `(action, object, source, flags, incoming-edge count)`, and directed
  producer/consumer type pairs. Features come from **all observed events**.
  Query features come from the same observed endpoint-rooted dependency DAG the
  verifier compares. One history contributes at most once per feature, avoiding
  a count advantage for long/dense histories. Score is a presence-weighted
  overlap, with weight `1 + floor(log2(floor((G+1)/(df+1))))`. This is an inverted
  index over individual facts, not a canonical history or answer signature.
* **Dependency shortlist (B):** typed event postings additionally retain
  `(gallery history, event index)`. Starting from raw query dependency nodes in
  chronological order, a relational semijoin retains candidate nodes whose
  canonical incoming writer edges point into their query parents' match sets.
  Observed first/last scoped roots must match the corresponding candidate
  boundaries. Intersect the two root projections to shortlist histories, then
  rank by A's score. The join does not enforce global injectivity, shared-source
  aliasing or mapped chronology. A regression deliberately admits an alias
  mismatch that the full verifier rejects. The index stores individual facts
  and real edges, never a precomputed whole-DAG signature.
* **HDC/A/B followed by verification:** pass only the first min(K,pool) histories
  to the unchanged `windowed::verify_dependencies`. Check every budgeted
  candidate, even after two acceptances, and return the first two accepted.
  Verification counts include failures. No fallthrough to unranked histories.
* **Dependency + HDC + verification:** B's actual pool is an eligibility mask
  for the same GPU HDC score; re-rank that pool and verify its top K. Empty
  shortlists remain empty. This tests whether HDC improves B's ordering.
* **Inverted top-64 + HDC + verification:** a second hybrid takes the first
  min(64,G) A candidates before HDC re-ranking. The fixed 64-candidate pool was
  declared before validation; it tests a more ambiguous exact shortlist when
  B has already reduced a query to two candidates. No test result tunes it.

Unverified rows return the first min(K,2) ranked candidates. Their candidate
recall@K can exceed final recall@2 because no verifier chooses later candidates.

## Scaling corpus

The original 16-gallery v3 collections are read unchanged and their archived
`w8_best` scores must match bit for bit. Separate nested scaling galleries use
16, 128 and 1,024 candidates, with a scaled G=16 control. Every size uses the
same eight queries and preserves existing histories byte for byte as gallery
size increases. All 256 families use the same object IDs, first/last writer
types, flags and action multiset: a ten-operation writer chain begins/ends in
action 4, with eight intermediate actions drawn from permutations of
`3,3,3,3,4,4,16,16`. A seed-shuffled 256 of the 420 unique permutations are used.
These are real action orders, not encoded family identifiers. Dependencies
connect actual versions of object 2. Order variants swap the final rewrite and
approval; independent filler replicas provide two relevant histories.

Queries use only four families; other families are hard negatives sharing
objects, action counts and endpoints, with different ordered dependency chains.
Family/order/replica and observed-byte FNV checksums are emitted solely for
evaluation/audit. There are no duplicate whole observed histories intended as
negatives. Unit tests assert exact relevance agrees with the unchanged verifier
on the controlled corpus and paired action multisets are identical.

Profiles preserve sparse filler, many unrelated significant objects, dense
target writers, and dense target operations across distant version links.
`dense_target_far` has 256-/2,048-step approval/rewrite gaps at lengths
4,096/32,768 and intervening significant operations on the same object; writer
chain links also span distant regions. Far cases use only the two lengths
supported by the original v3 suite. Other profiles run 32/1,024/4,096/32,768.
Validation seed 2027 and test seed 42 use different dictionaries and filler.
Settings are fixed before validation and locked before any held-out test runs.

These corpora test retrieval of observed ordered graphs, not arbitrary policy
outcomes or causal generalization. Filler flags remain observable because the
old encoder/verifier use them. Clean synthetic flags/edges can make an exact
index unrealistically effective; test richer provenance before generalizing.

## CUDA and binary representation

CUDA targets `sm_75`. A warp encodes each selected window with the existing
1,024-bit representation. Four warps compare a query's four contexts with eight
candidate contexts using XOR, popcount and warp reductions. A parallel reduction
emits the stable top 16 without the old 256-candidate scratch limit. Eligibility
masks skip non-shortlisted similarities; launch/ranking overhead still scales
with gallery size. Single-query paths encode their raw observed contexts on
device before comparison. Every GPU vector, full candidate score and ranking is
compared with scalar CPU calculations outside timed intervals.

No new persisted temporal binary format is introduced. Input is unchanged
HDBTMP01: 32-byte `Event`, 12-byte dependency `Edge`; oracle records are separate.
The transient raw context is eight 32-byte events followed by a uint32 count
(260 bytes). Device fingerprint cache is `[history][8][32]` uint32 words; query
fingerprints are `[query][4][32]`. Scores are `[query][gallery]` uint32 and ranks
`[query][16]` uint32, with `0xffffffff` for ineligible/missing results. Host/device
fields use the existing little-endian x64 ABI. Step/dependency/reserved fields
are retained in contexts but not read as HDC features.

## Timings, memory and counters

Construction includes scanning complete gallery histories and preparing selected
contexts for HDC, all exact-index postings for CPU, allocations, H2D copies and
GPU encoding. CUDA context initialization, file loading/generation, output
writing and CPU/GPU reference checking are excluded. The HDC support cache is
not charged as though the entire v3 window index had been rebuilt.

`total_query_ms` is measured wall time for a query from its complete, already
loaded observed history: scope selection, raw-context H2D, GPU encoding,
similarities, top-16 retrieval, D2H, shortlist processing and budgeted graph
verification as applicable. All reference assertions run after the timer stops.
`gpu_similarity_ms` and `gpu_topk_ms` are CUDA-event times, not wall latency.
CPU shortlist and verifier time are separately timed. There is no early stopping
after the second accepted candidate. Means use the configured repeats after GPU
warmup. CPU and GPU operate sequentially to avoid concurrent benchmark contention.

Batched throughput uses eight pre-encoded query fingerprints, one transfer per
batch, no graph verification and no CPU query scanning. It is explicitly a
different workload from full single-query latency. Reuse of resident gallery
indices is assumed in both cases.

GPU memory reports allocated buffer payload and temporary construction peak,
excluding CUDA driver/context overhead. CPU index memory is a vector-capacity
and map-overhead estimate, not an allocator audit; complete raw histories and
sampled process working set are reported separately. Inverted-index bytes include
the typed node postings needed by B because the structures share storage.
Raw CPU bytes use allocated vector capacities for the gallery and queries;
construction event/edge counts refer to gallery histories only.
Semijoin counters count examined typed-node postings, writer-edge membership
checks and inverted postings; they do not count every edge read during parent
canonicalization or query-plan traversal. Existing verifier counters count successfully mapped nodes/edges before
failure, a lower bound on all comparisons; they are not total process CPU work.
`score_ties_at_cutoff` is the full HDC distance-group size at K (including the
cutoff history), and zero for non-HDC rows; hybrids use the original full-gallery
group for audit, not just eligible histories.
