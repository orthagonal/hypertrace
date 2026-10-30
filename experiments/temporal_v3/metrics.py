"""Held-out relevance scoring; never imported by retrieval executables."""
import csv
from pathlib import Path


def rows(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def collection(data, results):
    catalog = {int(r["history_id"]): (r["family"], r["order"]) for r in rows(data / "catalog.csv")}
    roles = {int(r["history_id"]): int(r["role"]) for r in rows(data / "roles.csv")}
    score_rows, rank_rows = rows(results / "scores.csv"), rows(results / "ranks.csv")
    dep_rows = rows(results / "dependency_ranks.csv")
    methods = sorted({r["method"] for r in score_rows})
    output = []
    for method in methods:
        for q, role in roles.items():
            if not role:
                continue
            relevant = {c for c in roles if not roles[c] and catalog[q] == catalog[c]}
            assert relevant
            scores = {int(r["candidate_id"]): int(r["score"]) for r in score_rows if r["method"] == method and int(r["query_id"]) == q}
            assert set(scores) == {c for c in roles if not roles[c]}
            ranked = [int(r["candidate_id"]) for r in sorted(
                (r for r in rank_rows if r["method"] == method and int(r["query_id"]) == q), key=lambda r: int(r["k"]))]
            assert ranked == sorted(scores, key=lambda c: (scores[c], c))
            variants = [(method, 0, ranked)]
            if method.startswith("w"):
                for budget in (2, 4, 8, 16):
                    verified = [int(r["candidate_id"]) for r in sorted((r for r in dep_rows if r["method"] == method and
                        int(r["query_id"]) == q and int(r["budget"]) == budget), key=lambda r: int(r["k"]))]
                    assert set(verified) <= set(ranked[:budget])
                    variants.append(("dep_" + method, budget, verified))
            for name, budget, returned in variants:
                top = returned[:1]
                tied = {c for c in (returned if budget else scores) if scores[c] == min(scores[x] for x in returned)} if returned else set()
                positive = [scores[c] for c in relevant]
                negative = [scores[c] for c in scores if c not in relevant]
                row = {"method": name, "base_method": method, "budget": budget, "query_id": q,
                       "gallery": len(scores), "relevant": len(relevant), "hit1": int(bool(top) and top[0] in relevant),
                       "wrong1": sum(c not in relevant for c in top), "coverage1": int(bool(top)),
                       "tie_size": len(tied), "tie_aware_hit1": len(tied & relevant) / len(tied) if tied else 0,
                       "separation_hamming_equivalent": (sum(negative) / len(negative) - sum(positive) / len(positive)) / 1024 if negative else 0,
                       "candidate_recall": len(set(ranked[:budget]) & relevant) / len(relevant) if budget else 1}
                for k in (1, 2):
                    row[f"recall{k}"] = len(set(returned[:k]) & relevant) / len(relevant)
                    row[f"incorrect{k}"] = sum(c not in relevant for c in returned[:k])
                output.append(row)
    return output
