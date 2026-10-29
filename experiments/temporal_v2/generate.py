"""Isolated v2 retrieval corpus, using the unchanged HDBTMP01 observed format.

Catalog family/order/oracle fields are scoring metadata only. Executables read
events.bin and roles.csv; they never read catalog.csv.
"""
import argparse
import csv
import json
import random
import struct
from collections import Counter
from pathlib import Path

PRIVATE, EXTERNAL, APPROVAL, DERIVED, SENSITIVE, LOCAL, UNTRUSTED, FILLER = (1 << i for i in range(8))
WRITES = {1, 2, 3, 4, 8, 15, 16}
PROFILES = ("sparse", "dense_unrelated", "dense_target")
FAMILIES = ((4, 4), (3, 16), (16, 3), (4, 16))


def wire(actions):
    events, edges = [], []
    writers, sensitive, approved, approval_event = {}, {}, {}, {}
    for step, (action, obj, source, flags) in enumerate(actions):
        begin = len(edges)
        if source:
            edges.append((step, writers[source], source))
        elif action not in WRITES and obj in writers:
            edges.append((step, writers[obj], obj))
        if action in WRITES:
            writers[obj] = step
            sensitive[obj] = bool(flags & SENSITIVE)
            approved[obj] = False
            approval_event.pop(obj, None)
        elif action == 11:
            approved[obj], approval_event[obj] = True, step
        elif action in (5, 13, 14) and obj in approval_event:
            edges.append((step, approval_event[obj], obj))
        events.append((action, obj, source, flags, step, begin, len(edges) - begin, 0))
    label = int(bool(actions) and sensitive.get(2, False) and not approved.get(2, False))
    return events, edges, label


def history(n, family, order, replica, profile, seed):
    # The seed excludes order: paired histories have the exact same action multiset.
    rng = random.Random(seed + family * 100003 + replica * 1009 + n * 13)
    first, last = FAMILIES[family]
    actions = []
    known = {1, 2}
    for step in range(n):
        obj = 1000 + rng.randrange(32)
        action = rng.choice((6, 15, 17))
        fields = (action, 5000 + obj - 1000, 0, FILLER | (LOCAL if action == 15 else 0))
        if n >= 8 and step >= 2 and step < n - 4 and step % 2 == 0 and profile != "sparse":
            obj = 2 if profile == "dense_target" else obj
            action = rng.choice((4, 8, 11, 16, 5, 13, 14))
            if obj not in known or (obj == 2 and action == 8):
                action = 4
            if action in (4, 16):
                fields = (action, obj, 1, FILLER | DERIVED | SENSITIVE)
            elif action == 8:
                fields = (action, obj, 0, FILLER | PRIVATE | SENSITIVE)
            elif action == 11:
                fields = (11, obj, 0, FILLER | APPROVAL)
            else:
                fields = (action, obj, 0, FILLER | SENSITIVE | (EXTERNAL if action == 13 else LOCAL))
            known.add(obj)
        actions.append(fields)
    if n >= 8:
        actions[0] = (1, 1, 0, PRIVATE | SENSITIVE)
        actions[1] = (first, 2, 1, DERIVED | SENSITIVE)
        # Target-heavy noise precedes the final order-dependent pair, so both
        # outcomes remain controlled even after thousands of payload rewrites.
        a, b = (n - 4, n - 3) if profile == "dense_target" else (2 + (n - 8) // 3, 3 + 2 * (n - 8) // 3)
        rewrite, approve = (last, 2, 1, DERIVED | SENSITIVE), (11, 2, 0, APPROVAL)
        actions[a], actions[b] = (approve, rewrite) if order else (rewrite, approve)
        actions[n - 2] = (5, 2, 0, LOCAL | SENSITIVE)
        actions[n - 1] = (13, 2, 0, EXTERNAL | UNTRUSTED | SENSITIVE)
    return wire(actions)


def write_dataset(path, histories, pair_ids=None):
    count = len(histories)
    ne, nd = sum(len(h[0]) for h in histories), sum(len(h[1]) for h in histories)
    assert ne <= 2_000_000 and nd <= 4_000_000
    event_offset = 64 + count * 32
    edge_offset = event_offset + ne * 32
    label_offset = edge_offset + nd * 12
    with path.open("wb") as f:
        f.write(b"HDBTMP01" + struct.pack("<14I", 1, 64, count, ne, nd, 32, 12, 32, 64,
                                         event_offset, edge_offset, label_offset, 16, 0))
        eb = db = 0
        for i, (events, edges, label) in enumerate(histories):
            pair = i // 2 if pair_ids is None else pair_ids[i]
            f.write(struct.pack("<8I", i, pair, eb, len(events), db, len(edges), len(events), 0))
            eb += len(events)
            db += len(edges)
        for events, _, _ in histories:
            for event in events:
                f.write(struct.pack("<8I", *event))
        for _, edges, _ in histories:
            for edge in edges:
                f.write(struct.pack("<3I", *edge))
        for i, (_, _, label) in enumerate(histories):
            f.write(struct.pack("<4I", i, label, 2, label))


def generate(directory, lengths=(32, 1024, 4096, 32768), families=4, replicas=3, seed=42):
    assert 1 <= families <= 4 and replicas >= 2
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for n in lengths:
        for profile in PROFILES:
            name = f"{profile}_{n}"
            folder = directory / name
            folder.mkdir(exist_ok=True)
            entries = [(f, o, r) for f in range(families) for o in (0, 1) for r in range(replicas)]
            random.Random(seed + n).shuffle(entries)  # IDs/tie order do not favor an outcome.
            histories = [history(n, f, o, r, profile, seed) for f, o, r in entries]
            for f in range(families):
                for r in range(replicas):
                    a, b = [history(n, f, o, r, profile, seed) for o in (0, 1)]
                    assert Counter(e[:4] for e in a[0]) == Counter(e[:4] for e in b[0])
                    if n >= 8:
                        assert (a[2], b[2]) == (0, 1)
            write_dataset(folder / "events.bin", histories, [family * replicas + replica for family, _, replica in entries])
            with (folder / "roles.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(("history_id", "role"))
                writer.writerows((i, int(replica == replicas - 1)) for i, (_, _, replica) in enumerate(entries))
            with (folder / "catalog.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(("history_id", "family", "order", "replica", "oracle_label"))
                writer.writerows((i, family, order, replica, histories[i][2]) for i, (family, order, replica) in enumerate(entries))
            manifest.append({"name": name, "profile": profile, "length": n, "histories": len(histories)})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", default="32,1024,4096,32768")
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate(args.out, tuple(map(int, args.lengths.split(","))), args.families, args.replicas, args.seed)
    print(f"Generated {len(manifest)} v2 retrieval collections -> {args.out}")
