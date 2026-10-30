"""Produce compact CSVs and a measured report from validated checkpoints only."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode = True
from run import read_rows, digest


def mean(rows, name):
    return statistics.fmean(float(r[name]) for r in rows)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def report(root, out):
    out.mkdir(parents=True, exist_ok=True)
    summary, construction, separation, provenance = [], [], [], {}
    for checkpoint in sorted(root.glob("*/**/complete.json")):
        saved = json.loads(checkpoint.read_text()); folder = checkpoint.parent
        for name, sha in saved["outputs"].items():
            assert digest(folder / name) == sha, f"Modified output: {folder / name}"
        case = saved["inputs"]
        base = {"phase": case["phase"], "suite": case["suite"], "gallery": case["gallery"], "profile": case["case"]["profile"], "length": case["case"]["length"]}
        cr = read_rows(folder / "construction.csv")[0]
        construction.append({**cr, **base})
        groups = defaultdict(list)
        for row in read_rows(folder / "queries.csv"):
            groups[(row["strategy"], row["budget"])].append(row)
        for (strategy, budget), rows in groups.items():
            cell = {**base, "strategy": strategy, "budget": int(budget), "query_count": len(rows)}
            for key in rows[0]:
                if key not in ("strategy", "query_id", "budget"):
                    cell[key] = mean(rows, key)
            cell["false_matches_total"] = sum(int(r["false_matches"]) for r in rows)
            cell["abstentions_total"] = sum(int(r["abstention"]) for r in rows)
            cell["verifications_total"] = sum(int(r["verifications"]) for r in rows)
            summary.append(cell)
        truth = {r["history_id"]: (r["family"], r["order"]) for r in read_rows(folder / "histories.csv")}
        qgroups = defaultdict(list)
        for r in read_rows(folder / "scores.csv"):
            qgroups[r["query_id"]].append(r)
        for q, scores in qgroups.items():
            positive = min(int(r["score"]) for r in scores if truth[q] == truth[r["candidate_id"]]) / 1024
            negative = min(int(r["score"]) for r in scores if truth[q] != truth[r["candidate_id"]]) / 1024
            best = min(int(r["score"]) for r in scores)
            separation.append({**base, "query_id": q, "closest_relevant_bits": positive, "closest_negative_bits": negative,
                               "negative_minus_relevant_bits": negative - positive, "best_distance_group_size": sum(int(r["score"]) == best for r in scores)})
        provenance[str(folder.relative_to(root))] = digest(checkpoint)
    assert summary, "No completed measurements"
    write_csv(out / "summary.csv", summary)
    write_csv(out / "construction.csv", construction)
    write_csv(out / "separation.csv", separation)
    (out / "provenance.json").write_text(json.dumps({"checkpoints": provenance, "report_generator_sha256": digest(Path(__file__)), "settings": json.loads((root / "settings.json").read_text()),
        "hardware": (root / "hardware.txt").read_text(), "immutable": json.loads((root / "immutable_verified.json").read_text())}, indent=2))
    test = [r for r in summary if r["phase"] == "test"]

    def cell(suite, gallery, profile, length, strategy, k):
        matches = [r for r in test if r["suite"] == suite and r["gallery"] == gallery and r["profile"] == profile and r["length"] == length and r["strategy"] == strategy and r["budget"] == k]
        return matches[0] if matches else None

    def pct(v): return f"{100 * v:.1f}%"
    def val(r, key, scale=1): return f"{float(r[key]) * scale:.3f}" if r else "—"

    text = ["# Temporal Memory v3.1: candidate retrieval benchmark", "",
            "Measured on " + (root / "hardware.txt").read_text().splitlines()[1] + ". Validation seed 2027; held-out seed 42. No settings were selected on test results. The original frozen method remains `w8_best` (four query contexts, eight gallery contexts).", "",
            "## Findings", "",
            "Two gallery histories are relevant per query. Candidate recall is surviving relevant histories divided by two; final recall@2 counts relevant **returned** histories divided by two. Every K=1 result is capped at 50%. Rejected histories and abstentions contribute zero recall.", "",
            "### How far can the HDC budget shrink?", "",
            "Original v3 gallery, 32,768 events; candidate recall (HDC + verification has the same recall where verification is sound):", "",
            "| Profile | K=1 | K=2 | K=4 | K=8 | K=16 | Smallest tested K retaining both matches |",
            "|---|---:|---:|---:|---:|---:|---|"
            ]
    for profile in ("sparse", "dense_unrelated", "dense_target", "dense_target_far"):
        cells = [cell("original", 16, profile, 32768, "hdc_verified", k) for k in (1,2,4,8,16)]
        if all(cells):
            k = next((str(r["budget"]) for r in cells if r["candidate_recall"] == 1), ">16")
            text.append(f"| {profile} | " + " | ".join(pct(r["candidate_recall"]) for r in cells) + f" | {k} |")
    text += ["", "The dense failure remains a candidate-generation failure. Raising the verification budget cannot restore a relevant history that the shortlist excluded.", "",
             "### Can exact indexing rank more reliably, and does HDC help after it?", "",
             "Held-out 32,768-event histories, K=2 (candidate recall / final recall@2):", "",
             "| Suite / G / profile | HDC + verify | Inverted + verify | Dependency join + verify | Dependency + HDC + verify | Inverted top-64 + HDC + verify |",
             "|---|---:|---:|---:|---:|---:|"]
    selections = [("original",16,p) for p in ("sparse","dense_unrelated","dense_target","dense_target_far")]
    selections += [("scaled",g,"dense_target") for g in (16,128,1024)]
    for suite,g,p in selections:
        cells = [cell(suite,g,p,32768,s,2) for s in ("hdc_verified","inverted_verified","dependency_verified","hybrid_verified","hybrid_inverted_verified")]
        if all(cells):
            text.append(f"| {suite} / {g} / {p} | " + " | ".join(f"{pct(r['candidate_recall'])} / {pct(r['final_recall2'])}" for r in cells) + " |")
    dep = [r for r in test if r["strategy"] == "dependency_verified" and r["budget"] == 2]
    hybrid = [r for r in test if r["strategy"] == "hybrid_verified" and r["budget"] == 2]
    text += ["", f"Dependency joins retain {pct(mean(dep, 'candidate_recall'))} of relevant candidates at K=2 across the held-out suite. The dependency/HDC hybrid retains {pct(mean(hybrid, 'candidate_recall'))}. "
             "Where the join already yields the two relevant histories, HDC has no ranking headroom. The additional inverted-top-64 hybrid tests an ambiguous shortlist; it is frozen at 64, with no test-driven threshold.", "",
             "### Scaling and cost", "",
             "Scaled `dense_target`, 32,768 events. Single-query wall time includes scanning the complete observed query, window encoding, host/device transfers, ranking and verification. Resident indices are reused; original file loading and one-time CUDA context startup are excluded.", "",
             "| Gallery | HDC build ms | Exact build ms | GPU similarity µs | HDC + verify K=16 ms | Dependency + verify K=2 ms | Hybrid K=2 ms | HDC candidate recall K=16 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    inverted = [r for r in test if r["strategy"]=="inverted_verified" and r["budget"]==2]
    inverted_hybrid = [r for r in test if r["strategy"]=="hybrid_inverted_verified" and r["budget"]==2]
    # State both occasional gains and the paired overall effect; no ranking is tuned.
    improvements=regressions=0
    for a in inverted:
        b=cell(a["suite"],a["gallery"],a["profile"],a["length"],"hybrid_inverted_verified",2)
        improvements+=b["final_recall2"]>a["final_recall2"]
        regressions+=b["final_recall2"]<a["final_recall2"]
    insertion=text.index("### Scaling and cost")
    text[insertion:insertion]=[f"On the weaker inverted shortlist, HDC improves {improvements}/{len(inverted)} paired test collections and hurts {regressions}/{len(inverted)} at K=2. Mean final recall changes from {pct(mean(inverted,'final_recall2'))} to {pct(mean(inverted_hybrid,'final_recall2'))}; occasional gains do not establish a reliable benefit.", ""]
    for g in (16,128,1024):
        build = next((r for r in construction if r["phase"]=="test" and r["suite"]=="scaled" and r["gallery"]==g and r["profile"]=="dense_target" and r["length"]==32768),None)
        a,b,c = [cell("scaled",g,"dense_target",32768,s,k) for s,k in (("hdc_verified",16),("dependency_verified",2),("hybrid_verified",2))]
        if build and a and b and c:
            text.append(f"| {g} | {val(build,'hdc_index_wall_ms')} | {val(build,'exact_index_ms')} | {val(a,'gpu_similarity_ms',1000)} | {val(a,'total_query_ms')} | {val(b,'total_query_ms')} | {val(c,'total_query_ms')} | {pct(a['candidate_recall'])} |")
    text += ["", "| Gallery | Exact shortlist µs | Verify shortlisted µs | Raw history MiB | Added exact index MiB (estimate) | GPU resident MiB | GPU batched pre-encoded queries/s |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for g in (16,128,1024):
        build = next((r for r in construction if r["phase"]=="test" and r["suite"]=="scaled" and r["gallery"]==g and r["profile"]=="dense_target" and r["length"]==32768),None)
        b=cell("scaled",g,"dense_target",32768,"dependency_verified",2)
        if build and b:
            text.append(f"| {g} | {val(b,'shortlist_ms',1000)} | {val(b,'verification_ms',1000)} | {val(build,'raw_cpu_bytes',1/2**20)} | {val(build,'exact_index_estimated_bytes',1/2**20)} | {val(build,'gpu_resident_bytes',1/2**20)} | {val(build,'batch_queries_per_second')} |")
    text += ["", "GPU similarity kernel time is not end-to-end latency. The batched figure amortizes transfers over eight already encoded queries and excludes query preparation and verification; it must not be compared directly with full single-query wall time. CSVs include kernel top-K time, complete query latency, graph work and CPU working-set measurements.", "",
             "### GPU advantage and next direction", "",
             "Assess retrieval cost at matched recall. On the original dense gallery, compare HDC K=16 with dependency K=2. On larger hard-negative galleries, an HDC budget that loses matches is not an equivalent faster solution. "
             "These results do not establish a useful end-to-end GPU retrieval advantage over the dependency index. GPU HDC stores a small support cache and provides fast similarity kernels, but those properties alone do not establish useful candidate reduction.", "",
             "Pursue conventional object/version dependency indexing, with the unchanged full verifier retained for aliasing and chronology. Test its robustness on less synthetic dependency graphs before changing the HDC architecture. Do not start v4 based on fingerprint differences or kernel throughput alone.", "",
             "## False matches, abstentions and work", "",
             "At K=2, held-out totals across all completed original and scaled cases:", "",
             "| Strategy | Mean candidate recall | Mean final recall@2 | False returned histories | Zero-result queries | Full verifications |",
             "|---|---:|---:|---:|---:|---:|"]
    h=cell("original",16,"dense_target",32768,"hdc_verified",16)
    e=cell("original",16,"dense_target",32768,"dependency_verified",2)
    if h and e:
        insertion=text.index("### GPU advantage and next direction")
        text[insertion:insertion]=[f"At matched 100% recall on the original long dense-target gallery, HDC K=16 takes {val(h,'total_query_ms')} ms/query with {h['verifications']:.0f} full verifications; the dependency index K=2 takes {val(e,'total_query_ms')} ms/query with {e['verifications']:.0f} verifications.", ""]
    for strategy in ("random_only","hdc_only","inverted_only","dependency_only","random_verified","hdc_verified","inverted_verified","dependency_verified","hybrid_verified","hybrid_inverted_verified"):
        rows=[r for r in test if r["strategy"]==strategy and r["budget"]==2]
        text.append(f"| {strategy} | {pct(mean(rows,'candidate_recall'))} | {pct(mean(rows,'final_recall2'))} | {sum(r['false_matches_total'] for r in rows)} | {sum(r['abstentions_total'] for r in rows)} | {sum(r['verifications_total'] for r in rows)} |")
    text += ["", "## Verification and limitations", "",
             f"All {len(construction)} collections passed independent scalar CPU/GPU fingerprint, similarity-score and top-16 ranking comparisons, including masked hybrid searches. Original archived `w8_best` scores are compared exactly for all original collections. Prior source/test hashes are unchanged. Regression coverage includes empty/partial windows, 1,024-way ties, 32,768-step histories, future-event rejection, metadata isolation and a source-version alias that survives the semijoin but fails full verification.", "",
             "The original G=16 benchmark is preserved. Larger galleries use a separate nested suite, with a scaled G=16 control: 256 distinct ordered writer chains made from the same action multiset, identical endpoint types and shared object IDs. Two independently filled replicas are relevant; opposite-order replicas are hard negatives. Results from the scaled suite must not be attributed solely to gallery size by comparing them directly with the original grammar.", "",
             "Only four query families (eight queries) are used per collection, and the two phase seeds are not confidence intervals. Synthetic Filler flags and clean writer edges make the conventional join easier than uncertain real provenance. The join is a necessary-condition filter, not full graph isomorphism; it may admit alias/chronology errors on richer data. The verifier only compares its existing endpoint-rooted dependency DAG, not arbitrary whole-history policy equivalence. Raw histories are kept in full; only frozen HDC support contexts are cached. Neither HDC nor retrieval recall is exact policy enforcement.", "",
             "Construction excludes validation/reference checks. GPU memory is allocated buffer payload, not driver/context overhead; CPU exact-index bytes are an allocation estimate, with sampled process working set separately reported. Existing verifier node counts record successfully mapped nodes before failure, a lower bound on all comparisons; semijoin node/posting/edge work is counted separately. GPU top-K always emits the same 16-rank prefix, so K varies verification budget rather than the ranking algorithm. Latencies are arithmetic means of the configured repeated runs on this machine, not service-level guarantees.", "",
             "Full profile × length × gallery × budget results, both validation and test: [summary.csv](summary.csv). Construction/memory/batched throughput: [construction.csv](construction.csv). Hamming separation: [separation.csv](separation.csv). Source/settings/output hashes and hardware: [provenance.json](provenance.json). Per-query outcomes and candidate acceptance traces remain in the reproducible runner output under `target/candidate_benchmark`."]
    (out / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Wrote measured report and {len(summary)} aggregate CSV rows to {out}")


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); report(a.results,a.out)
