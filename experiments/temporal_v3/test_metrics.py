"""Hand-calculated recall, abstention, ties, candidate coverage and freeze checks."""
import csv
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True
from metrics import collection
from selection import algorithm_hashes, validate_freeze


def write(path, header, records):
    with path.open("w", newline="") as file:
        writer = csv.writer(file); writer.writerow(header); writer.writerows(records)


def main():
    Path("target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v3_metrics_", dir="target") as temp:
        root = Path(temp)
        write(root / "catalog.csv", ("history_id", "family", "order", "replica", "oracle_label"),
              ((0, 0, 0, 0, 0), (1, 0, 0, 1, 0), (2, 0, 1, 0, 1), (3, 0, 0, 2, 0)))
        write(root / "roles.csv", ("history_id", "role"), ((0, 0), (1, 0), (2, 0), (3, 1)))
        write(root / "scores.csv", ("method", "query_id", "candidate_id", "score"), (("w8_best", 3, 0, 10), ("w8_best", 3, 1, 10), ("w8_best", 3, 2, 10)))
        write(root / "ranks.csv", ("method", "query_id", "k", "candidate_id", "score"), (("w8_best", 3, 1, 0, 10), ("w8_best", 3, 2, 1, 10), ("w8_best", 3, 3, 2, 10)))
        write(root / "dependency_ranks.csv", ("method", "budget", "query_id", "k", "candidate_id", "score"), (("w8_best", 4, 3, 1, 0, 10), ("w8_best", 4, 3, 2, 1, 10)))
        result = collection(root, root)
        approximate = next(r for r in result if not r["budget"])
        assert approximate["recall1"] == 0.5 and approximate["recall2"] == 1 and approximate["hit1"] == 1
        assert approximate["tie_size"] == 3 and approximate["tie_aware_hit1"] == 2 / 3
        abstain = next(r for r in result if r["budget"] == 2)
        assert abstain["coverage1"] == abstain["wrong1"] == abstain["recall2"] == 0
        accepted = next(r for r in result if r["budget"] == 4)
        assert accepted["recall2"] == 1 and accepted["incorrect2"] == 0 and accepted["candidate_recall"] == 1
        frozen = root / "frozen.json"
        frozen.write_text(json.dumps({"phase": "validation", "algorithm_sha256": algorithm_hashes()}))
        validate_freeze(frozen)
        frozen.write_text(json.dumps({"phase": "test", "algorithm_sha256": algorithm_hashes()}))
        try: validate_freeze(frozen)
        except AssertionError: pass
        else: raise AssertionError("test data was allowed to select settings")
    print("PASS: hand-calculated v3 recall, abstention, tie accounting and validation freeze")


if __name__ == "__main__": main()
