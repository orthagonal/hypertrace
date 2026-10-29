"""End-to-end diagnostic contract tests using small independent edge-case records.

Usage: python tests/diagnostics_integration_test.py --cpu <exe> [--gpu <exe>]
GPU mode also exercises --verify and compares every diagnostic artifact with CPU.
"""
import argparse
import csv
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ARTIFACTS = ("records.csv", "disagreements.csv", "groups.csv", "ablations.csv", "summary.txt")
SPLITS = ("train", "validation", "test", "long-test")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_dataset(path, records):
    with path.open("wb") as f:
        f.write(b"HDBSAFE1" + struct.pack("<4I", 1, len(records), 1024, 12))
        for row in records:
            f.write(struct.pack("<12I", *row))


def execute(exe, dataset, directory=None, gpu=False):
    args = [str(exe), str(dataset)]
    if gpu:
        args.append("--verify")
    if directory is not None:
        args += ["--diagnostics", str(directory)]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    if gpu:
        assert "PASS: GPU encoder, class prototypes, and every prediction match CPU." in result.stdout
    return result.stdout


def validate(directory, records):
    rows = read_csv(directory / "records.csv")
    examples = read_csv(directory / "disagreements.csv")
    groups = read_csv(directory / "groups.csv")
    ablations = read_csv(directory / "ablations.csv")
    assert len(rows) == len(records)
    assert len(ablations) == 4 * 24  # 20 individual removals + four controls per split.
    for i, (row, original) in enumerate(zip(rows, records)):
        assert int(row["record_index"]) == i
        assert row["split"] == SPLITS[original[0]]
        assert int(row["observed_length"]) == original[3]
        assert int(row["oracle_label"]) == original[1]
        assert row["scenario_type"] == row["violation_category"] == row["enforcement_outcome"] == "not_recorded"
        safe, unsafe, margin = map(int, (row["distance_safe"], row["distance_unsafe"], row["margin_safe_minus_unsafe"]))
        assert 0 <= safe <= 1024 and 0 <= unsafe <= 1024
        assert margin == safe - unsafe
        assert int(row["hdc_prediction"]) == (unsafe < safe)
        flags = original[2]
        assert int(row["rules_prediction"]) == bool(flags & 2 and flags & 4 and not flags & 16 and flags & 32)
    expected_examples = [r for r in rows if r["hdc_prediction"] != r["rules_prediction"]]
    assert examples == expected_examples  # All disagreements, with no display truncation.
    assert Counter(r["disagreement"] for r in examples).keys() <= {"hdc_correct_rules_wrong", "rules_correct_hdc_wrong"}
    # Aggregate groups from exported rows, rather than trusting the report's counters.
    for g in groups:
        members = [r for r in rows if r["split"] == g["split"] and r[{
            "length": "length_bucket", "observed_cohort": "observed_cohort",
            "oracle_outcome": "oracle_outcome", "hdc_error": "hdc_error"
        }[g["dimension"]]] == g["group"]]
        assert len(members) == int(g["n"]) > 0
        for model in ("hdc", "rules"):
            correct = sum(r[f"{model}_prediction"] == r["oracle_label"] for r in members)
            assert abs(float(g[f"{model}_accuracy_pct"]) - 100 * correct / len(members)) < 0.0001
            for error in ("fp", "fn"):
                assert int(g[f"{model}_{error}"]) == sum(r[f"{model}_error"] == error.upper() for r in members)
        assert abs(float(g["mean_margin"]) - sum(int(r["margin_safe_minus_unsafe"]) for r in members) / len(members)) < 0.0001
    for split in SPLITS:
        count = sum(r["split"] == split for r in rows)
        for dimension in ("length", "observed_cohort", "oracle_outcome", "hdc_error"):
            assert sum(int(g["n"]) for g in groups if g["split"] == split and g["dimension"] == dimension) == count
        interventions = [a for a in ablations if a["split"] == split]
        assert len({a["intervention"] for a in interventions}) == 24
        for a in interventions:
            assert int(a["n"]) == count
            if count:
                assert abs(float(a["delta_pp"]) - 100 * (int(a["errors_fixed"]) - int(a["correct_predictions_broken"])) / count) < 0.0001
                assert int(a["prediction_flips"]) == int(a["errors_fixed"]) + int(a["correct_predictions_broken"])
            else:
                assert a["accuracy_pct"] == a["baseline_accuracy_pct"] == a["delta_pp"] == "NA"
    assert "prediction FP/FN are not enforcement failures" in (directory / "summary.txt").read_text()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path)
    args = parser.parse_args()
    args.cpu = args.cpu.resolve()
    args.gpu = args.gpu.resolve() if args.gpu else None
    # No validation rows: deliberately test empty evaluation splits.
    records = [
        (0, 0, 0, 10, *([6] * 8)),
        (0, 0, 2 | 4 | 16 | 32, 20, *([8] * 8)),
        (0, 1, 2 | 4 | 32, 30, *([3] * 8)),
        (0, 1, 2 | 4 | 32 | 64, 40, *([16] * 8)),
    ]
    lengths = (0, 9, 10, 16, 17, 32, 33, 64, 65, 127, 128, 256, 257, 512, 513)
    for i, length in enumerate(lengths):
        records.append((2, i % 2, (i * 293) & 4095, length, *range(3, 11)))
    records += [(3, 0, 38, 128, *range(3, 11)), (3, 1, 38, 512, *range(3, 11))]
    Path("target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="diagnostics_test_", dir="target") as temporary:
        base = Path(temporary)
        dataset = base / "edge_cases.bin"
        write_dataset(dataset, records)
        normal = execute(args.cpu, dataset)
        extended = execute(args.cpu, dataset, base / "cpu")
        assert normal == extended.split("\nHDC diagnostics:")[0] + "\nUse CPU/GPU" + extended.split("\nUse CPU/GPU", 1)[1]
        original = validate(base / "cpu", records)
        # Hold training fixed; alter all evaluation labels, lengths and splits.
        changed = [r if r[0] == 0 else (3 if r[0] == 2 else 2, 1-r[1], r[2], r[3]+1000, *r[4:]) for r in records]
        changed_data = base / "changed_metadata.bin"
        write_dataset(changed_data, changed)
        execute(args.cpu, changed_data, base / "changed")
        altered = validate(base / "changed", changed)
        for a, b in zip(original, altered):
            for field in ("hdc_prediction", "rules_prediction", "distance_safe", "distance_unsafe", "margin_safe_minus_unsafe"):
                assert a[field] == b[field], f"metadata leaked into {field}"
        if args.gpu:
            execute(args.gpu, dataset, base / "gpu", gpu=True)
            validate(base / "gpu", records)
            for name in ARTIFACTS:
                assert (base / "gpu" / name).read_bytes() == (base / "cpu" / name).read_bytes(), name
        invalid = subprocess.run([str(args.cpu), str(dataset), "--diagnostics"], capture_output=True)
        assert invalid.returncode != 0
        print("PASS: diagnostics artifacts, all length boundaries, empty splits, group accounting, metadata isolation and optional GPU parity")


if __name__ == "__main__":
    main()
