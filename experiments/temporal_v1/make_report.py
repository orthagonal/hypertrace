"""Rebuild a measured Markdown report from the experiment's CSV/binary artifacts."""
import argparse
import csv
import hashlib
import struct
from pathlib import Path


def rows(directory, name):
    with (directory / name).open(newline="") as f:
        return list(csv.DictReader(f))


def table(headers, values):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in values])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--cpu-results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    raw = args.dataset.read_bytes()
    assert raw[:8] == b"HDBTMP01"
    header = struct.unpack_from("<14I", raw, 8)
    deterministic = [f"history_{i}.mem" for i in range(header[2])] + [name + ".csv" for name in
        ("memory", "retrieval", "retrieved_events", "regions", "query_distances", "sensitivity", "correctness")]
    for name in deterministic:
        assert (args.cpu_results / name).read_bytes() == (args.results / name).read_bytes(), name
    memory, timings, retrieval, sensitivity, correctness = [rows(args.results, name + ".csv") for name in
        ("memory", "timings", "retrieval", "sensitivity", "correctness")]
    assert all(all(r[field] == "PASS" for field in ("leaf_fields_and_vectors", "all_internal_fields", "all_masks", "queries", "exact_search")) for r in correctness)
    assert len(correctness) == header[2] and sum(int(r["events"]) for r in correctness) == header[3]
    ids = sorted({int(r["history_id"]) for r in memory if int(r["history_id"]) % 2 == 0})
    mem_rows, time_rows, hit_rows, change_rows = [], [], [], []
    for history in ids:
        m = [r for r in memory if int(r["history_id"]) == history]
        n = int(m[0]["events"])
        sizes = {int(r["level"]): int(r["total_bytes"]) for r in m}
        mem_rows.append([history, n] + [f"{sizes.get(l, 0):,}" for l in range(4)] + [f"{sum(sizes.values()):,}"])
        t = [r for r in timings if int(r["history_id"]) == history and r["variant"] == "original"]
        def ms(kernel, level=None):
            matches = [float(r["median_ms"]) for r in t if r["kernel"] == kernel and (level is None or int(r["level"]) == level)]
            return f"{sum(matches):.6f}" if matches else "—"
        query_sum = sum(float(r["median_ms"]) for r in t if r["kernel"] == "compare_query" and int(r["level"]) > 0)
        search_sum = sum(float(r["median_ms"]) for r in t if r["kernel"] in ("descend_masks_private_read", "exact_leaf_filter_private_read"))
        time_rows.append([n, ms("encode_history", 0)] + [ms("build_temporal_level", l) for l in range(1, 4)] +
                         [f"{query_sum:.6f}", f"{search_sum:.6f}"])
        r = next(r for r in retrieval if int(r["history_id"]) == history and r["property"] == "private_read")
        hit_rows.append([n, r["exact_matches"], r["regions"], 0, 0])
        changes = {r["variant"]: r["root_hamming"] for r in sensitivity if int(r["history_id"]) == history and r["level"] == "0"}
        change_rows.append([n] + [changes.get(k, "NA") for k in
            ("paired_action_order", "controlled_reorder", "controlled_remove_approval", "controlled_gap")])
    repetitions = sorted({int(r["repeats"]) for r in timings})
    total_nodes = sum(int(r["internal_nodes"]) for r in correctness)
    hardware = (args.results / "summary.txt").read_text().splitlines()[0]
    collisions = [str(r[0]) for r in change_rows if r[1] == "0"]
    collision_text = ("In this run, the policy order pair(s) at length(s) " + ", ".join(collisions) +
                      " have root distance zero despite different policy outcomes.") if collisions else (
        "No root collision was observed in these policy pairs; that does not establish general preservation.")
    text = f"""# Temporal Memory v1 measured report

This isolated experiment uses {header[2]} full observed histories, {header[3]:,} events,
and {header[4]} separately stored exact dependency edges. Dataset SHA-256:
`{hashlib.sha256(raw).hexdigest()}`.
The original v0 encoder, trained prototypes, CUDA learner and benchmarks are unchanged.
There is no new classifier. See FORMAT.md for every binary field and encoding equation.

## Correctness

All {header[3]:,} GPU leaves and {total_nodes:,} internal nodes match the independent scalar
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

{table(["History", "Events", "L0", "L1", "L2", "L3", "Tree total"], mem_rows)}

Totals exclude input staging (32 bytes/event), query buffers, exact-search frontiers,
allocator overhead, and archived exact edges (12 bytes/edge). Edges are retained in
the dataset and memory file, not included in GPU HDC computations. Disk files also
have a 64-byte header and 24-byte descriptors per level. No full provenance graph
is claimed by masks.

## CUDA kernel measurements (milliseconds)

Hardware/implementation: {hardware}.
Measured with CUDA events, two warmups and repetition count(s)
{repetitions}. Cells are medians. Summary-query and private-search columns are **sums
of separate kernel medians**, not end-to-end latency. Allocation, CPU verification,
transfers and file I/O are excluded. Other display workloads and clock variation can
affect these samples; this is a reproducibility measurement, not a tuned benchmark.

{table(["Events", "Encode", "Build L1", "Build L2", "Build L3", "Query summaries", "Private search"], time_rows)}

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

{table(["Events", "Paired order", "Controlled order", "Remove approval", "Long gap"], change_rows)}

Local changes can be distinguishable at lower levels yet disappear after repeated
bundling. {collision_text} Thus a correct, order-sensitive implementation
does **not** imply a useful root representation. All aligned-level mean/max distances
are exported in sensitivity.csv. Removing an event can shift grouping boundaries;
moving it changes positional binding, not an independently modeled elapsed duration.
These few synthetic pairs do not establish general temporal reasoning or predictive
accuracy. Results for other seeds/pair counts must be measured, not assumed.

## Exact private-read retrieval

Both members of each pair have the same private-read locations. All eight property
queries are checked; this table illustrates the requested exhaustive private-read query.

{table(["Events", "Private reads found", "Internal regions", "False negatives", "False positives"], hit_rows)}

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
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote measured temporal report: {args.out}")


if __name__ == "__main__":
    main()
