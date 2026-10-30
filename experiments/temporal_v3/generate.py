"""Disjoint validation/test corpora, preserving the v2 protocol and adding far edges."""
import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import random
import sys
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("v2_generation_unchanged", Path(__file__).resolve().parents[1] / "temporal_v2/generate.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def far_history(n, gap, family, order, replica, seed):
    rng = random.Random(seed + family * 100003 + replica * 1009 + n * 13)
    first, last = v2.FAMILIES[family]
    actions = []
    for i in range(n):
        action = rng.choice((5, 12, 13, 14))
        actions.append((action, 2, 0, v2.FILLER | v2.SENSITIVE | (v2.EXTERNAL if action == 13 else v2.LOCAL)))
    actions[0] = (1, 1, 0, v2.PRIVATE | v2.SENSITIVE)
    actions[1] = (first, 2, 1, v2.DERIVED | v2.SENSITIVE)
    early, late = n // 4, n // 4 + gap + 1
    assert late < n - 2
    rewrite, approve = (last, 2, 1, v2.DERIVED | v2.SENSITIVE), (11, 2, 0, v2.APPROVAL)
    actions[early], actions[late] = (approve, rewrite) if order else (rewrite, approve)
    actions[-2] = (5, 2, 0, v2.LOCAL | v2.SENSITIVE)
    actions[-1] = (13, 2, 0, v2.EXTERNAL | v2.UNTRUSTED | v2.SENSITIVE)
    return v2.wire(actions)


def generate(directory, phase, lengths=(32, 1024, 4096, 32768), families=4):
    seed = 2027 if phase == "validation" else 42
    manifest = v2.generate(directory, lengths, families, 3, seed)
    for n, gap in ((4096, 256), (32768, 2048)):
        if n not in lengths:
            continue
        name = f"dense_target_far_{gap}_{n}"
        folder = directory / name
        folder.mkdir(exist_ok=True)
        entries = [(f, o, r) for f in range(families) for o in (0, 1) for r in range(3)]
        random.Random(seed + n + gap).shuffle(entries)
        histories = [far_history(n, gap, f, o, r, seed) for f, o, r in entries]
        for f in range(families):
            for r in range(3):
                a, b = [far_history(n, gap, f, o, r, seed) for o in (0, 1)]
                assert Counter(e[:4] for e in a[0]) == Counter(e[:4] for e in b[0])
                assert (a[2], b[2]) == (0, 1)
        v2.write_dataset(folder / "events.bin", histories, [f * 3 + r for f, _, r in entries])
        with (folder / "roles.csv").open("w", newline="") as file:
            writer = csv.writer(file); writer.writerow(("history_id", "role"))
            writer.writerows((i, int(r == 2)) for i, (_, _, r) in enumerate(entries))
        with (folder / "catalog.csv").open("w", newline="") as file:
            writer = csv.writer(file); writer.writerow(("history_id", "family", "order", "replica", "oracle_label"))
            writer.writerows((i, f, o, r, histories[i][2]) for i, (f, o, r) in enumerate(entries))
        manifest.append({"name": name, "profile": "dense_target_far", "length": n, "histories": len(entries), "gap": gap})
    for item in manifest:
        item["phase"], item["seed"] = phase, seed
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("validation", "test"), required=True)
    p.add_argument("--lengths", default="32,1024,4096,32768")
    p.add_argument("--families", type=int, default=4)
    p.add_argument("--frozen", type=Path)
    args = p.parse_args()
    if args.phase == "test":
        assert args.frozen, "held-out generation requires a validation freeze"
        from selection import validate_freeze
        validate_freeze(args.frozen)
    print(f"Generated {len(generate(args.out, args.phase, tuple(map(int,args.lengths.split(','))), args.families))} {args.phase} collections")
