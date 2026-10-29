"""Score retrieval using held-out catalog metadata, never supplied to CUDA/encoders."""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from test_integration import rows, parity

METHODS = ("v1_root", "v2_global", "v2_object_scoped", "exact_pattern")


def metrics(source, results):
    catalog = {int(r["history_id"]): (r["family"], r["order"], r["replica"]) for r in rows(source / "catalog.csv")}
    roles = {int(r["history_id"]): int(r["role"]) for r in rows(source / "roles.csv")}
    distances, ranks = rows(results / "distances.csv"), rows(results / "rankings.csv")
    output = []
    for method in METHODS:
        for query, role in roles.items():
            if not role:
                continue
            relevant = {c for c in roles if not roles[c] and catalog[c][:2] == catalog[query][:2]}
            assert relevant and all(catalog[c][2] != catalog[query][2] for c in relevant)
            scores = {int(r["candidate_id"]): int(r["distance"]) for r in distances
                      if r["method"] == method and int(r["query_id"]) == query}
            assert set(scores) == {c for c in roles if not roles[c]}
            found = sorted((int(r["k"]), int(r["candidate_id"])) for r in ranks
                           if r["method"] == method and int(r["query_id"]) == query)
            ranked = [candidate for _, candidate in found]
            assert ranked == sorted(scores, key=lambda c: (scores[c], c))[:5]
            best = min(scores.values())
            tied = {c for c, d in scores.items() if d == best}
            row = {"method": method, "query_id": query, "gallery": len(scores), "relevant": len(relevant),
                   "hit1": int(ranked[0] in relevant), "wrong1": int(ranked[0] not in relevant),
                   "tie_size": len(tied), "tie_aware_hit1": len(tied & relevant) / len(tied),
                   "zero_false_matches": sum(d == 0 and c not in relevant for c, d in scores.items())}
            for k in (1, 2, 5):
                hits = len(set(ranked[:k]) & relevant)
                row[f"recall{k}"] = hits / len(relevant)
                row[f"false{k}"] = len(ranked[:k]) - hits
            if method == "exact_pattern":
                assert all((d == 0) == (c in relevant) for c, d in scores.items()), "exact signature differs from relevance"
            output.append(row)
    return output


def table(headers, data):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in data])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    initial_catalog = rows(args.data / manifest[0]["name"] / "catalog.csv")
    initial_roles = rows(args.data / manifest[0]["name"] / "roles.csv")
    family_count = len({r["family"] for r in initial_catalog})
    gallery_count = sum(r["role"] == "0" for r in initial_roles)
    query_count = sum(r["role"] == "1" for r in initial_roles)
    gallery_replicas = len({r["replica"] for r in initial_catalog}) - 1
    recall_ceiling = 100 / gallery_replicas
    forced_false5 = max(0, min(5, gallery_count) - gallery_replicas)
    all_metrics, paired, resources, fingerprint_digests = [], [], [], []
    for item in manifest:
        source, output = args.data / item["name"], args.results / item["name"]
        parity(output / "cpu", output / "gpu")
        fingerprint_digests.append((item["name"], hashlib.sha256((source / "events.bin").read_bytes()).hexdigest()))
        for row in metrics(source, output / "gpu"):
            all_metrics.append({"profile": item["profile"], "length": item["length"], **row})
        catalog = rows(source / "catalog.csv")
        vectors = {(int(r["history_id"]), r["method"]): int(r["words_hex"], 16) for r in rows(output / "gpu" / "fingerprints.csv")}
        for a in catalog:
            if a["order"] != "0":
                continue
            b = next(r for r in catalog if r["family"] == a["family"] and r["replica"] == a["replica"] and r["order"] == "1")
            for method in METHODS[:3]:
                distance = (vectors[(int(a["history_id"]), method)] ^ vectors[(int(b["history_id"]), method)]).bit_count()
                paired.append({"profile": item["profile"], "length": item["length"], "method": method,
                               "first": int(a["history_id"]), "second": int(b["history_id"]), "hamming": distance})
        memory = rows(output / "gpu" / "memory.csv")
        timings = rows(output / "gpu" / "timings.csv")
        n = len(memory)
        sparse_time = sum(float(r["median_ms"]) for r in timings if r["kernel"].startswith("sparse_")) / n
        v1_time = sum(float(r["median_ms"]) for r in timings if r["kernel"].startswith("v1_")) / n
        exact_time = sum(float(r["median_ms"]) for r in timings if r["kernel"] == "exact_pattern_key") / n
        resources.append({"profile": item["profile"], "length": item["length"],
            "anchors": statistics.mean(int(r["anchors"]) for r in memory),
            "scoped_anchors": statistics.mean(int(r["scoped_anchors"]) for r in memory),
            "v1_tree_bytes": statistics.mean(int(r["v1_tree_bytes"]) for r in memory),
            "v2_index_bytes": statistics.mean(int(r["anchor_bytes"]) + int(r["v2_fingerprint_bytes"]) for r in memory),
            "scratch_upper_bound_bytes": statistics.mean(int(r["scratch_upper_bound_bytes"]) for r in memory),
            "v1_build_ms": v1_time, "v2_build_ms": sparse_time, "exact_key_ms": exact_time,
            "collection_score_ms": sum(float(r["median_ms"]) for r in timings if r["kernel"].startswith("collection_"))})
    for name, data in (("query_metrics.csv", all_metrics), ("paired_distances.csv", paired), ("resources.csv", resources)):
        with (args.results / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    (args.results / "dataset_hashes.json").write_text(json.dumps(dict(fingerprint_digests), indent=2))
    sections = []
    for profile in ("sparse", "dense_unrelated", "dense_target"):
        values = []
        for item in manifest:
            if item["profile"] != profile:
                continue
            columns = [item["length"]]
            for method in METHODS:
                group = [r for r in all_metrics if r["profile"] == profile and r["length"] == item["length"] and r["method"] == method]
                columns.append(f"{100*statistics.mean(r['recall1'] for r in group):.1f} / {100*statistics.mean(r['recall2'] for r in group):.1f}")
            values.append(columns)
        sections.append("### " + profile + "\n\n" + table(["Actions", "v1 root", "v2 global", "v2 scoped", "Exact pattern"], values))
    hamming_rows, false_rows = [], []
    for item in manifest:
        values = [item["profile"], item["length"]]
        for method in METHODS[:3]:
            group = [r["hamming"] for r in paired if r["profile"] == item["profile"] and r["length"] == item["length"] and r["method"] == method]
            values.append(f"{statistics.mean(group):.1f} [{min(group)},{max(group)}]")
        hamming_rows.append(values)
        if item["length"] == max(i["length"] for i in manifest):
            values = [item["profile"]]
            for method in METHODS:
                group = [r for r in all_metrics if r["profile"] == item["profile"] and r["length"] == item["length"] and r["method"] == method]
                values.append(f"{sum(r['wrong1'] for r in group)}/{len(group)}")
            false_rows.append(values)
    resource_rows = [[r["profile"], r["length"], f"{r['anchors']:.0f}/{r['scoped_anchors']:.0f}",
                      f"{r['v1_tree_bytes']/1024:.1f}", f"{r['v2_index_bytes']/1024:.1f}",
                      f"{r['v1_build_ms']:.3f}/{r['v2_build_ms']:.3f}"] for r in resources]
    first = args.results / manifest[0]["name"] / "gpu"
    hardware = (first / "summary.txt").read_text().splitlines()[0]
    repeats = sorted({int(r["repeats"]) for r in rows(first / "timings.csv")})
    longest = max(i["length"] for i in manifest)
    findings = []
    for profile in ("sparse", "dense_unrelated", "dense_target"):
        measured = []
        for method in METHODS:
            group = [r for r in all_metrics if r["profile"] == profile and r["length"] == longest and r["method"] == method]
            measured.append(f"{100*statistics.mean(r['recall2'] for r in group):.1f}%")
        findings.append(f"- {profile}, {longest:,} actions: recall@2 is " + " / ".join(measured) + " (v1 / global / scoped / exact).")
    text = f"""# Temporal Memory v2: Sparse Anchors — measured report

{hardware}. CUDA repetition counts: {repeats}; two warmups per kernel.
The original v0/v1 sources, full tree, tests and trained prototypes are unchanged.
Every selected anchor, exact writer version, event vector, fingerprint, all-pairs score
and GPU top-K result matches the independent scalar CPU reference. Full v1 trees are
kept in separate GPU buffers and checked unchanged after sparse construction.

{chr(10).join(findings)}

## Retrieval task

Each collection has {manifest[0]['histories']} independently generated histories. {family_count}
motif families each have two approval/rewrite orders with opposite later outcomes,
{gallery_replicas} gallery replicas and one held-out query replica ({gallery_count} gallery, {query_count} queries). Object
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

Cells are **recall@1 / recall@2 (%)**. Each query has {gallery_replicas} relevant gallery histories,
so perfect top-1 retrieval gives recall@1={recall_ceiling:.1f}%.
All methods use the same candidates and deterministic (distance, history-ID) ordering;
IDs were shuffled independently of outcome. Full recall@5, wrong matches, zero-distance
false matches and tie-aware top-1 success are in query_metrics.csv.

{chr(10).join(sections)}

## False top-1 matches at the longest tested length

Counts are wrong temporal patterns / queries. Fixed top-K always returns K available
candidates, even for the exact baseline: at K=5, at least {forced_false5} are irrelevant because
only {gallery_replicas} relevant candidates exist. Interpret false@5 with that constraint.

{table(["Profile", "v1", "v2 global", "v2 scoped", "Exact"], false_rows)}

## Opposite-order pair Hamming distances

Mean [min,max], out of 1,024 bits, over all matched replica/family pairs. These distances
are diagnostics; the recall tables above determine whether they support retrieval.

{table(["Profile", "Actions", "v1 root", "v2 global", "v2 scoped"], hamming_rows)}

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

{table(["Profile", "Actions", "Anchors all/scoped", "v1 KiB", "Added v2 KiB", "Build ms v1/v2"], resource_rows)}

## Interpretation and limits

The measured sparse case preserves the requested patterns across long irrelevant gaps.
Object scoping also succeeds when many significant events concern unrelated objects.
However, the measured dense-target case remains unsolved by v2: flat majority bundling
still dilutes differences when many anchors concern the same queried object. This is
an experimental limitation, not a correctness failure. Distinct fingerprints alone
are insufficient; the held-out retrieval results expose that distinction.

This corpus is a controlled, single-seed, {family_count}-family retrieval task with explicit
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
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote sparse retrieval report: {args.out}")


if __name__ == "__main__":
    main()
