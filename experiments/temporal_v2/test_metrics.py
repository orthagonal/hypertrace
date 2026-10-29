"""Hand-calculated retrieval checks: recall is not top-1 success; ties are audited."""
import csv
import sys
import tempfile
from pathlib import Path
sys.dont_write_bytecode = True
from make_report import metrics, METHODS


def write(path, header, data):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def main():
    Path("target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sparse_metrics_", dir="target") as temp:
        root = Path(temp)
        write(root / "catalog.csv", ("history_id", "family", "order", "replica", "oracle_label"),
              ((0, 0, 0, 0, 0), (1, 0, 0, 1, 0), (2, 0, 1, 0, 1), (3, 0, 0, 2, 0)))
        write(root / "roles.csv", ("history_id", "role"), ((0, 0), (1, 0), (2, 0), (3, 1)))
        distances, ranks = [], []
        for method in METHODS:
            values = (0, 0, 1024) if method == "exact_pattern" else (0, 0, 0)
            distances.extend((method, 3, c, d) for c, d in enumerate(values))
            ranks.extend((method, 3, c + 1, c, d) for c, d in enumerate(values))
        write(root / "distances.csv", ("method", "query_id", "candidate_id", "distance"), distances)
        write(root / "rankings.csv", ("method", "query_id", "k", "candidate_id", "distance"), ranks)
        for row in metrics(root, root):
            assert row["recall1"] == 0.5 and row["hit1"] == 1 and row["wrong1"] == 0
            assert row["recall2"] == row["recall5"] == 1 and row["false2"] == 0 and row["false5"] == 1
            assert row["zero_false_matches"] == (0 if row["method"] == "exact_pattern" else 1)
            assert row["tie_aware_hit1"] == (1 if row["method"] == "exact_pattern" else 2 / 3)
    print("PASS: hand-calculated recall@K, false matches, eligibility and tie-aware retrieval metrics")


if __name__ == "__main__":
    main()
