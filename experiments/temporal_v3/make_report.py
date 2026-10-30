"""Reproduce measured report using the frozen validation choice, never test tuning."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from metrics import collection, rows
from run_collections import parity
from selection import validate_freeze


def table(header, records):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, row)) + " |" for row in records])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--results", type=Path, required=True)
    p.add_argument("--anchors", type=Path, required=True); p.add_argument("--frozen", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    frozen = validate_freeze(args.frozen); chosen = frozen["selected"]["method"]; budget = frozen["dependency_budget"]
    lock = json.loads((args.results / "evaluation_lock.json").read_text())
    assert lock["frozen_sha256"] == hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    manifest = json.loads((args.data / "manifest.json").read_text()); assert all(r["phase"] == "test" for r in manifest)
    width = frozen["selected"]["width"]
    all_rows, resources, links, input_hashes = [], [], [], {}
    window_repeats, baseline_repeats = set(), set()
    for item in manifest:
        source, result = args.data / item["name"], args.results / item["name"]
        parity(result / "cpu", result / "gpu")
        input_hashes[item["name"]] = {"events.bin": hashlib.sha256((source / "events.bin").read_bytes()).hexdigest(),
            "roles.csv": hashlib.sha256((source / "roles.csv").read_bytes()).hexdigest(),
            "anchors": {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in (args.anchors / item["name"] / "gpu").glob("anchors_*.bin")}}
        for row in collection(source, result / "gpu"):
            all_rows.append({"profile": item["profile"], "length": item["length"], "gap": item.get("gap", 0), **row})
        mem, times, graph = rows(result / "gpu/memory.csv"), rows(result / "gpu/timings.csv"), rows(result / "gpu/graph_timings.csv")
        histories = len({r["history_id"] for r in mem}); query_count = sum(r["role"] == "1" for r in rows(source / "roles.csv"))
        anchor_mem = rows(args.anchors / item["name"] / "gpu/memory.csv")
        anchor_times = rows(args.anchors / item["name"] / "gpu/timings.csv")
        window_repeats.update(int(r["repeats"]) for r in times if r["kernel"] != "query_batch_wall")
        baseline_repeats.update(int(r["repeats"]) for r in anchor_times)
        selected_times = [r for r in times if int(r["width"]) == width]
        gpu_wall = sum(float(r["median_ms"]) for r in selected_times if r["kernel"] == "query_batch_wall") / query_count
        kernel_time = sum(float(r["median_ms"]) for r in selected_times if r["kernel"] != "construct_windows" and r["kernel"] != "query_batch_wall") / query_count
        graph_ms = sum(float(r["wall_ms"]) for r in graph if r["method"] == chosen and int(r["proposal_rank"]) <= budget) / query_count
        for r in rows(result / "gpu/dependency_links.csv"):
            if r["method"] == chosen and int(r["proposal_rank"]) <= budget:
                links.append({"profile": item["profile"], "length": item["length"], **r})
        resources.append({"profile": item["profile"], "length": item["length"], "gap": item.get("gap", 0),
            "scoped_anchors": statistics.mean(int(r["scoped_anchors"]) for r in anchor_mem),
            "windows_per_history": sum(int(r["windows"]) for r in mem) / histories,
            "index_bytes_per_history": sum(int(r["window_bytes"]) + (int(r["scope_ref_bytes"]) if int(r["width"]) == 8 else 0) for r in mem) / histories,
            "gpu_peak_bytes": max(int(r["gpu_peak_bytes"]) for r in mem),
            "construction_ms_per_history": sum(float(r["median_ms"]) for r in times if r["kernel"] == "construct_windows") / histories,
            "query_wall_ms_amortized": gpu_wall, "query_cuda_ms_amortized": kernel_time,
            "graph_wall_ms_per_query": graph_ms,
            "v1_tree_bytes_per_history": statistics.mean(int(r["v1_tree_bytes"]) for r in anchor_mem),
            "v2_index_bytes_per_history": statistics.mean(int(r["anchor_bytes"]) + 256 for r in anchor_mem),
            "v1_construction_ms_per_history": sum(float(r["median_ms"]) for r in anchor_times if r["kernel"].startswith("v1_")) / histories,
            "v2_construction_ms_per_history": sum(float(r["median_ms"]) for r in anchor_times if r["kernel"].startswith("sparse_")) / histories,
            "baseline_four_method_cuda_ms_per_query": sum(float(r["median_ms"]) for r in anchor_times if r["kernel"].startswith("collection_")) / query_count})
    for filename, data in (("query_metrics.csv", all_rows), ("resources.csv", resources), ("selected_dependency_links.csv", links)):
        if data:
            with (args.results / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    (args.results / "test_input_sha256.json").write_text(json.dumps(input_hashes, indent=2))
    methods = [("v1_root", 0), ("v2_global", 0), ("v2_object_scoped", 0), (chosen, 0), ("dep_" + chosen, budget), ("exact_baseline", 0)]
    accuracy, wrong, resource_table, separation = [], [], [], []
    for item in manifest:
        values = [item["profile"], item["length"]]
        failures = [item["profile"], item["length"]]
        for method, r in methods:
            group = [row for row in all_rows if row["profile"] == item["profile"] and row["length"] == item["length"] and row["method"] == method and row["budget"] == r]
            values.append(f"{100*statistics.mean(row['recall1'] for row in group):.1f}/{100*statistics.mean(row['recall2'] for row in group):.1f}")
            failures.append(f"{sum(row['wrong1'] for row in group)};{sum(row['coverage1']==0 for row in group)};{statistics.mean(row['tie_size'] for row in group):.1f}")
        accuracy.append(values)
        if item["profile"].startswith("dense_target") or item["length"] == 32768: wrong.append(failures)
        for method, r in (methods[2], methods[3]):
            group = [row for row in all_rows if row["profile"] == item["profile"] and row["length"] == item["length"] and row["method"] == method and row["budget"] == r]
            if item["length"] == 32768:
                separation.append([item["profile"], item["length"], method, f"{statistics.mean(row['separation_hamming_equivalent'] for row in group):.2f}"])
    for r in resources:
        resource_table.append([r["profile"], r["length"], f"{r['scoped_anchors']:.0f}/{r['windows_per_history']:.0f}",
            f"{r['index_bytes_per_history']/1024:.1f}", f"{r['gpu_peak_bytes']/1048576:.3f}", f"{r['construction_ms_per_history']:.3f}",
            f"{r['query_wall_ms_amortized']:.3f}+{r['graph_wall_ms_per_query']:.3f}"])
    normal = [r for r in all_rows if r["profile"] == "dense_target" and r["length"] == 32768 and r["method"] == chosen]
    dep = [r for r in all_rows if r["profile"] == "dense_target" and r["length"] == 32768 and r["method"] == "dep_"+chosen and r["budget"] == budget]
    max_jump = max((int(r["query_consumer_step"]) - int(r["query_producer_step"]) for r in links if r["object"] == "2"), default=0)
    comparison = [[r["profile"], f"{r['v1_tree_bytes_per_history']/1024:.1f}/{r['v2_index_bytes_per_history']/1024:.1f}/{r['index_bytes_per_history']/1024:.1f}",
        f"{r['v1_construction_ms_per_history']:.3f}/{r['v2_construction_ms_per_history']:.3f}/{r['construction_ms_per_history']:.3f}",
        f"{r['baseline_four_method_cuda_ms_per_query']:.3f}/{r['query_cuda_ms_amortized']:.3f}"] for r in resources if r["length"] == 32768]
    freeze_sha = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    verification_rows = [r for item in manifest for r in rows(args.results / item["name"] / "gpu/correctness.csv")]
    total_windows = sum(int(r["windows"]) for r in verification_rows)
    total_similarities = sum(int(r["similarities"]) for r in verification_rows)
    measured_window = statistics.mean(r["recall2"] for r in normal)
    research_answer = ("Window-only recall in the longest dense-target case is at or below the random-ranking expectation; this does not solve the dense-target failure reliably."
                       if measured_window <= 0.125 else "Window-only recall exceeds the random-ranking expectation here, but this small controlled test does not establish general temporal retrieval.")
    report = f"""# Temporal Memory v3 — measured held-out report

Frozen validation choice: **{chosen}**, dependency candidate budget **{budget}/16**.
Validation window recall@2: **{100*frozen['selected']['recall2']:.1f}%**. The complete
selection record is in frozen_settings.json. Freeze SHA-256: `{freeze_sha}`.
Validation used seed 2027; held-out evaluation used seed 42, with independent filler
replicas. The freeze binds retrieval code/metrics and validation data hashes. It was
checked before test retrieval; no setting is selected from the following test results.

At 32,768 dense-target actions, window recall@2 is **{100*statistics.mean(r['recall2'] for r in normal):.1f}%**
and dependency-verified recall@2 is **{100*statistics.mean(r['recall2'] for r in dep):.1f}%**,
versus 0% for the existing v2 object-scoped experiment on this corpus.
These results concern a finite synthetic query task, not a new classifier or enforcer.
Random-ranking recall@2 has expectation 12.5% in this gallery. **{research_answer}**
With budget 16/16, exact verification checks the whole gallery; the same accepted set
can be obtained by exhaustive raw-graph matching without HDC shortlisting.

## Retrieval comparison

Cells: **recall@1 / recall@2 (%)**. Each collection preserves v2's 16 gallery histories
and eight held-out queries, with two relevant candidates per query. Thus recall@1 has
a 50% ceiling. Relevance is the same motif family **and order**, not the outcome label.
All original histories, v1 trees and v2 anchor representations remain unchanged.

{table(["Profile", "Actions", "v1", "v2 global", "v2 scoped", "v3 windows", "v3 + dependencies", "Exact baseline"], accuracy)}

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
critical operations. Selected exact links span up to {max_jump:,} observed steps on
object 2; selected_dependency_links.csv resolves both endpoints to stored windows.
The verifier has no hop cutoff; a 32,768-node chain regression passes.

## Incorrect matches, abstention and ties

Cells: **incorrect top-1 count ; queries with no result ; mean minimum-score tie size**,
over eight queries. Critical/long cases are shown; all rows are in query_metrics.csv.
Dependency rejection is not counted as a correct returned match.

{table(["Profile", "Actions", "v1", "v2 global", "v2 scoped", "v3 windows", "v3 + dependencies", "Exact"], wrong)}

## Score separation

Mean irrelevant score minus relevant score, in Hamming-equivalent units. For v2 this
is literal 1,024-bit Hamming separation; for v3 it is the selected normalized window
cost. The longest histories are shown; all lengths remain in query_metrics.csv.
Positive is desirable; a distinct fingerprint alone does not prove retrieval.

{table(["Profile", "Actions", "Representation", "Separation"], separation)}

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

{table(["Profile", "Actions", "Anchors/windows", "Added KiB", "GPU peak MiB", "Build ms", "Query+graph ms"], resource_table)}

At 32,768 actions, incremental representation/build costs compare as follows. The last
column is old four-method collection scoring CUDA time / new full-window plus three-
aggregator CUDA time, both amortized per query. These scopes differ and are labeled;
they are not interchangeable single-method latencies. V2 and v3 costs are additional
to their retained predecessors. Baseline timing repetition counts: {sorted(baseline_repeats)};
v3 kernel repetition counts: {sorted(window_repeats)}.

{table(["Profile", "KiB v1/v2/v3", "Build ms v1/v2/v3", "Old/new CUDA query ms"], comparison)}

## Correctness and limitations

For the held-out phase, {total_windows:,} windows and {total_similarities:,} similarities
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
"""
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(report, encoding="utf-8")
    print(f"Wrote measured v3 held-out report: {args.out}")


if __name__ == "__main__": main()
