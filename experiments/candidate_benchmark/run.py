"""Bounded, resumable validation-then-test runner. No retrieval settings are fitted."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = {
    "version": "candidate-v3.1-1", "hdc": "w8_best", "width": 8,
    "query_contexts": 4, "gallery_contexts": 8, "budgets": [1, 2, 4, 8, 16],
    "validation_seed": 2027, "test_seed": 42, "random_seed": 173,
    "inverted": "presence-weighted primitive node/object/one-edge postings; integer log2 IDF",
    "dependency": "typed node/writer-edge semijoin with observed endpoint restrictions",
    "hybrid_inverted_pool": 64,
    "final_limit": 2, "relevance": "same family and order; two gallery replicas",
    "scaled_core": "10 writer-linked actions; 420 unique multiset permutations; 256 used",
    "tuning": "none; settings declared before validation; test never selects settings",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def source_hashes():
    files = [p for p in HERE.iterdir() if p.suffix in (".h", ".cuh", ".cu", ".py")]
    for name in ("temporal_v1", "temporal_v2", "temporal_v3"):
        files += [p for p in (HERE.parent / name).iterdir() if p.suffix in (".h", ".cu", ".py", ".json")]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def immutable_snapshot():
    files = []
    for name in ("src", "include", "cuda", "tests", "experiments/temporal_v1", "experiments/temporal_v2", "experiments/temporal_v3"):
        files += [p for p in (ROOT / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [p for p in (ROOT / "scripts").glob("*.ps1") if p.name != "run_candidate_benchmark.ps1"]
    files += [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def check_rows(folder):
    rows = read_rows(folder / "queries.csv")
    assert len(rows) == 8 * 10 * 5
    details = read_rows(folder / "candidates.csv")
    groups = {}
    for r in details:
        groups.setdefault((r["strategy"], r["query_id"], r["budget"]), []).append(r)
    for r in rows:
        k = int(r["budget"])
        ds = groups.get((r["strategy"], r["query_id"], r["budget"]), [])
        assert len(ds) <= k
        assert int(r["candidate_hits"]) == sum(int(x["relevant"]) for x in ds)
        returned = [x for x in ds if x["returned"] == "1"]
        assert len(returned) <= 2
        hits = sum(int(x["relevant"]) for x in returned)
        assert int(r["final_hits"]) == hits
        assert float(r["candidate_recall"]) == int(r["candidate_hits"]) / 2
        assert float(r["final_recall2"]) == hits / 2
        assert int(r["false_matches"]) == len(returned) - hits
        assert int(r["abstention"]) == int(not returned)
        assert int(r["partial"]) == int(len(returned) < 2)
        if k == 1:
            assert float(r["candidate_recall"]) <= .5
        if "verified" in r["strategy"]:
            assert int(r["verifications"]) == len(ds)
            assert int(r["rejected"]) == sum(x["accepted"] == "0" for x in ds)
        else:
            assert int(r["verifications"]) == 0
        assert float(r["total_query_ms"]) >= float(r["verification_ms"])
    for strategy in {r["strategy"] for r in rows}:
        for q in {r["query_id"] for r in rows}:
            series = sorted((r for r in rows if r["strategy"] == strategy and r["query_id"] == q), key=lambda r: int(r["budget"]))
            hits = [int(r["candidate_hits"]) for r in series]
            assert hits == sorted(hits), "candidate budget changed the underlying ranking"
    return rows


def original_parity(folder, old):
    if not old.is_file():
        raise AssertionError(f"Original v3 score archive is required: {old}")
    expected = {(r["query_id"], r["candidate_id"]): int(r["score"]) for r in read_rows(old) if r["method"] == "w8_best"}
    actual = {(r["query_id"], r["candidate_id"]): int(r["score"]) for r in read_rows(folder / "scores.csv")}
    assert actual == expected, "frozen v3 ranking changed"


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    before = immutable_snapshot()
    (out / "immutable_before.json").write_text(json.dumps(before, indent=2))
    frozen = json.loads((HERE.parent / "temporal_v3/frozen_settings.json").read_text())
    assert frozen["selected"]["method"] == "w8_best"
    for name, sha in frozen["algorithm_sha256"].items():
        assert digest(HERE.parent / "temporal_v3" / name) == sha, f"Original v3 freeze changed: {name}"
    source = source_hashes()
    settings = {"config": CONFIG, "sources": source, "repeats": args.repeats, "lengths": args.lengths, "galleries": args.galleries}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv"], capture_output=True, text=True)
    (out / "hardware.txt").write_text(hardware.stdout + hardware.stderr + "\n" + sys.version + "\n")
    phases = ("validation", "test")
    timings = []
    for phase in phases:
        if phase == "test":
            lock = json.loads((out / "validation_lock.json").read_text())
            assert lock["fingerprint"] == fingerprint
            assert source_hashes() == source, "code changed after validation"
        data = ROOT / "target/temporal_v3" / phase / "data"
        assert (data / "manifest.json").is_file(), "Run the existing v3 experiment first; original data are never overwritten"
        cases = []
        for item in json.loads((data / "manifest.json").read_text()):
            if item["length"] not in args.lengths:
                continue
            cases.append(("original", 16, item))
            for gallery in args.galleries:
                cases.append(("scaled", gallery, item))
        for ordinal, (suite, gallery, item) in enumerate(cases, 1):
            name = item["name"]
            folder = out / phase / suite / f"g{gallery}" / name
            folder.mkdir(parents=True, exist_ok=True)
            inputs = {"suite": suite, "gallery": gallery, "case": item, "phase": phase, "fingerprint": fingerprint}
            if suite == "original":
                inputs["data_sha256"] = {n: digest(data / name / n) for n in ("events.bin", "roles.csv", "catalog.csv")}
                if phase == "validation":
                    assert inputs["data_sha256"]["events.bin"] == frozen["validation_sha256"][name]["events.bin"]
            complete = folder / "complete.json"
            resume = False
            if args.resume and complete.exists():
                saved = json.loads(complete.read_text())
                resume = saved["inputs"] == inputs and all(digest(folder / n) == sha for n, sha in saved["outputs"].items())
            print(f"[{phase} {ordinal}/{len(cases)}] {suite} G={gallery} {name}" + (" (verified checkpoint)" if resume else ""), flush=True)
            start = time.perf_counter()
            if not resume:
                command = [str(args.exe.resolve()), "--out", str(folder), "--repeats", str(args.repeats)]
                if suite == "original":
                    command += ["--original", str(data / name)]
                else:
                    command += ["--gallery", str(gallery), "--length", str(item["length"]), "--profile", item["profile"], "--seed", str(CONFIG[f"{phase}_seed"])]
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                (folder / "run.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                check_rows(folder)
                if suite == "original":
                    original_parity(folder, ROOT / "target/temporal_v3" / phase / "results" / name / "gpu/scores.csv")
                (folder / "case.json").write_text(json.dumps(inputs, indent=2))
                names = ("queries.csv", "scores.csv", "histories.csv", "construction.csv", "candidates.csv", "case.json")
                complete.write_text(json.dumps({"inputs": inputs, "outputs": {n: digest(folder / n) for n in names}}, indent=2))
            timings.append({"phase": phase, "suite": suite, "gallery": gallery, "name": name, "wall_seconds": time.perf_counter() - start, "resumed": int(resume)})
        if phase == "validation":
            (out / "validation_lock.json").write_text(json.dumps({"fingerprint": fingerprint, "config": CONFIG, "selection": "none; test uses identical settings", "checkpoints": {str(p.relative_to(out)): digest(p) for p in (out / phase).rglob("complete.json")}}, indent=2))
    assert immutable_snapshot() == before, "previous implementation or tests changed"
    (out / "immutable_verified.json").write_text(json.dumps({"files": len(before), "unchanged": True}, indent=2))
    with (out / "run_times.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=timings[0]); writer.writeheader(); writer.writerows(timings)
    print(f"PASS: bounded benchmark complete; {len(timings)} collections; original source hashes unchanged", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--lengths", type=lambda v: list(map(int, v.split(","))), default=[32, 1024, 4096, 32768])
    p.add_argument("--galleries", type=lambda v: list(map(int, v.split(","))), default=[16, 128, 1024])
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    assert a.repeats > 0 and set(a.galleries) <= {16, 128, 1024}
    run(a)
