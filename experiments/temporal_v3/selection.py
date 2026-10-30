"""Select on validation only; freeze settings and retrieval-code hashes before test."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from metrics import collection

ALGORITHMS = ("reference.h", "experiment.h", "gpu.cu", "cpu.cpp", "generate.py", "metrics.py", "selection.py", "run_collections.py")


def algorithm_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ALGORITHMS}


def validate_freeze(path):
    frozen = json.loads(path.read_text())
    assert frozen["phase"] == "validation" and frozen["algorithm_sha256"] == algorithm_hashes(), "retrieval/metric code changed after validation freeze"
    return frozen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    assert manifest and all(item["phase"] == "validation" for item in manifest)
    all_rows, hashes = [], {}
    for item in manifest:
        source, result = args.data / item["name"], args.results / item["name"] / "gpu"
        hashes[item["name"]] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ("events.bin", "catalog.csv")}
        all_rows.extend(collection(source, result))
    summary = []
    for width in (8, 16, 32):
        for strategy in ("best", "normalized_top2", "ordered"):
            method = f"w{width}_{strategy}"
            rows = [r for r in all_rows if r["method"] == method]
            summary.append({"method": method, "width": width, "strategy": strategy,
                            "recall2": statistics.mean(r["recall2"] for r in rows),
                            "hit1": statistics.mean(r["hit1"] for r in rows)})
    strategy_order = {"best": 0, "normalized_top2": 1, "ordered": 2}
    chosen = min(summary, key=lambda r: (-r["recall2"], -r["hit1"], r["width"], strategy_order[r["strategy"]]))
    budgets = []
    for budget in (2, 4, 8, 16):
        rows = [r for r in all_rows if r["method"] == "dep_" + chosen["method"] and r["budget"] == budget]
        budgets.append({"budget": budget, "recall2": statistics.mean(r["recall2"] for r in rows)})
    selected_budget = min(budgets, key=lambda r: (-r["recall2"], r["budget"]))["budget"]
    frozen = {"phase": "validation", "seed": 2027, "selected": chosen, "dependency_budget": selected_budget,
              "query_slots": 4, "history_context_slots": 8,
              "selection_rule": "maximize mean validation window recall@2, then hit@1; ties smaller width then best/top2/ordered. Choose smallest graph budget achieving maximal validation recall@2 for that frozen method.",
              "ablations": summary, "graph_budgets": budgets, "validation_sha256": hashes,
              "algorithm_sha256": algorithm_hashes()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frozen, indent=2))
    print(f"FROZEN validation choice: {chosen['method']}, dependency budget {selected_budget}")


if __name__ == "__main__":
    main()
