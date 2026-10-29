"""Independent sparse binary/selection/provenance/retrieval tests. Standard library only."""
import argparse
import csv
import hashlib
import struct
import subprocess
import tempfile
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import generate


def run(command, success=True):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result.stdout


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def input_histories(path):
    raw = path.read_bytes()
    header = struct.unpack_from("<14I", raw, 8)
    histories = []
    for i in range(header[2]):
        d = struct.unpack_from("<8I", raw, 64 + i * 32)
        events = [struct.unpack_from("<8I", raw, header[9] + (d[2] + j) * 32) for j in range(d[3])]
        edges = [struct.unpack_from("<3I", raw, header[10] + (d[4] + j) * 12) for j in range(d[5])]
        histories.append((events, edges))
    return raw, header, histories


def significant(e):
    action, _, _, flags = e[:4]
    return bool((action in (1, 2, 8) or flags & 1) and flags & 16 or
                action in (3, 4, 5, 11, 12, 13, 14, 16) or flags & (2 | 4 | 8))


def hash32(value):
    value = ((value ^ (value >> 16)) * 0x7feb352d) & 0xffffffff
    value = ((value ^ (value >> 15)) * 0x846ca68b) & 0xffffffff
    return value ^ (value >> 16)


def event_vector(e):
    values = list(e[:3]) + [(e[3] >> bit) & 1 for bit in range(8)]
    vector = []
    for word in range(32):
        tokens = [hash32(((tag * 0x9e3779b9) ^ value ^ (word * 0x85ebca6b) ^ 0xa5f152bd) & 0xffffffff)
                  for tag, value in enumerate(values, 1)]
        vector.append(sum((sum((token >> bit) & 1 for token in tokens) >= 6) << bit for bit in range(32)))
    return tuple(vector)


def validate_index(path, original):
    raw = path.read_bytes()
    assert raw[:8] == b"HDBANC02"
    h = struct.unpack_from("<14I", raw, 8)
    assert h[:2] == (2, 64) and h[7] == 176 and h[9:11] == (12, 64) and h[13] == 104
    events, edges = original
    assert h[3] == len(events) and h[8] == len(edges)
    assert h[11] == 64 + h[4] * 176 and h[12] == h[11] + len(edges) * 12
    assert len(raw) == h[12] + 104 * 4
    selected = [(i, e) for i, e in enumerate(events) if significant(e)]
    assert len(selected) == h[4]
    assert h[6] == sum(e[1] == h[5] for _, e in selected)
    versions, latest = [], {}
    for e in events:
        action, obj, source, _, step = e[:5]
        source_version = latest.get(source, 0xffffffff) if source else 0xffffffff
        if action in (1, 2, 3, 4, 8, 15, 16) and obj:
            latest[obj] = step
        versions.append((latest.get(obj, 0xffffffff) if obj else 0xffffffff, source_version))
    for ordinal, (index, e) in enumerate(selected):
        anchor = struct.unpack_from("<44I", raw, 64 + ordinal * 176)
        assert anchor[:8] == e and anchor[8:12] == (index, *versions[index], ordinal)
        assert all(edge[0] == e[4] and edge[1] < e[4] for edge in edges[e[5]:e[5] + e[6]])
        # Every vector is checked by the independent C++ reference. Also sample
        # vectors against Python equations so serialization cannot hide an error.
        if ordinal < 3 or ordinal == len(selected) - 1:
            assert anchor[12:] == event_vector(e)
    archived = [struct.unpack_from("<3I", raw, h[11] + i * 12) for i in range(len(edges))]
    assert archived == edges
    tail = struct.unpack_from("<104I", raw, h[12])
    return [tail[i * 32:(i + 1) * 32] for i in range(3)], tail[96:]


def validate(directory, dataset, roles_path):
    _, _, histories = input_histories(dataset)
    data = [validate_index(directory / f"anchors_{i}.bin", h) for i, h in enumerate(histories)]
    roles = {int(r["history_id"]): int(r["role"]) for r in rows(roles_path)}
    distances = rows(directory / "distances.csv")
    ranks = rows(directory / "rankings.csv")
    for row in distances:
        q, c = int(row["query_id"]), int(row["candidate_id"])
        assert roles[q] == 1 and roles[c] == 0 and q != c
        method = ("v1_root", "v2_global", "v2_object_scoped", "exact_pattern").index(row["method"])
        expected = (0 if data[q][1] == data[c][1] else 1024) if method == 3 else sum(
            (a ^ b).bit_count() for a, b in zip(data[q][0][method], data[c][0][method]))
        assert expected == int(row["distance"])
    for method in ("v1_root", "v2_global", "v2_object_scoped", "exact_pattern"):
        for q, role in roles.items():
            if not role:
                continue
            candidates = sorted((int(r["distance"]), int(r["candidate_id"])) for r in distances
                                if r["method"] == method and int(r["query_id"]) == q)
            found = sorted((int(r["k"]), int(r["distance"]), int(r["candidate_id"])) for r in ranks
                           if r["method"] == method and int(r["query_id"]) == q)
            assert [(d, c) for _, d, c in found] == candidates[:5]
    return data


def parity(cpu, gpu):
    for file in cpu.iterdir():
        if file.name not in ("summary.txt", "timings.csv"):
            assert file.read_bytes() == (gpu / file.name).read_bytes(), file.name


def execute(cpu, gpu, folder, output="results"):
    dataset, roles = folder / "events.bin", folder / "roles.csv"
    cpu_out = folder / (output + "_cpu")
    run([cpu, dataset, "--roles", roles, "--out", cpu_out])
    validate(cpu_out, dataset, roles)
    if gpu:
        gpu_out = folder / (output + "_gpu")
        run([gpu, dataset, "--roles", roles, "--out", gpu_out, "--repeats", "1"])
        validate(gpu_out, dataset, roles)
        parity(cpu_out, gpu_out)
    return cpu_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--gpu", type=Path)
    args = p.parse_args()
    cpu, gpu = args.cpu.resolve(), args.gpu.resolve() if args.gpu else None
    Path("target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sparse_test_", dir="target") as temporary:
        base = Path(temporary)
        manifest = generate.generate(base / "corpus", (32,), 2, 3, 42)
        for item in manifest:
            folder = base / "corpus" / item["name"]
            output = execute(cpu, gpu, folder)
            catalog = {int(r["history_id"]): (r["family"], r["order"]) for r in rows(folder / "catalog.csv")}
            for r in rows(output / "rankings.csv"):
                if r["method"] == "exact_pattern" and r["k"] == "1":
                    assert catalog[int(r["query_id"])] == catalog[int(r["candidate_id"])], "exact baseline uses wrong relevance"
            raw, h, _ = input_histories(folder / "events.bin")
            mutated = bytearray(raw)
            for i in range(h[2]):
                label = struct.unpack_from("<I", raw, h[11] + i * 16 + 4)[0]
                struct.pack_into("<3I", mutated, h[11] + i * 16 + 4, 1 - label, 99, 77)
            (folder / "events.bin").write_bytes(mutated)
            changed = execute(cpu, gpu, folder, "changed")
            parity(output, changed)  # Includes anchors, v1 tree, keys, fingerprints and rankings.
        # Empty/singleton, prefix-scan boundaries, and dense significant streams.
        edge = base / "edge"
        edge.mkdir()
        histories = []
        for n in (0, 1, 31, 32, 33, 256, 257, 1025, 32768):
            histories.append(generate.history(n, 0, 0, 0, "dense_target", 123))
        histories.append(generate.wire(((15, 9, 0, generate.LOCAL), (5, 9, 0, generate.LOCAL), (16, 2, 9, generate.DERIVED))))
        generate.write_dataset(edge / "events.bin", histories)
        with (edge / "roles.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("history_id", "role"))
            writer.writerows((i, int(i % 2 == 1)) for i in range(len(histories)))
        execute(cpu, gpu, edge)
        raw, h, _ = input_histories(edge / "events.bin")
        broken = bytearray(raw)
        struct.pack_into("<I", broken, h[9] + 16, 1)  # First nonempty event at cutoff=1.
        bad = edge / "future.bin"
        bad.write_bytes(broken)
        run([cpu, bad, "--roles", edge / "roles.csv", "--out", edge / "invalid"], success=False)
        print("PASS: sparse/dense selection, exact versions/edges, independent vector samples, GPU top-K, ties, no outcome leakage, 32768-event dense histories and future rejection")


if __name__ == "__main__":
    main()
