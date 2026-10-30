"""Independent binary/window/score checks and outcome-leakage regression."""
import argparse
import array
import csv
import importlib.util
import itertools
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
sys.dont_write_bytecode = True
import generate
from run_collections import parity


def run(command, success=True):
    result = subprocess.run([str(c) for c in command], text=True, capture_output=True)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result.stdout


def rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def mix(v):
    v = ((v ^ (v >> 16)) * 0x7feb352d) & 0xffffffff
    v = ((v ^ (v >> 15)) * 0x846ca68b) & 0xffffffff
    return v ^ (v >> 16)


def fingerprint(anchors):
    vector = []
    for w in range(32):
        bound = []
        for i, a in enumerate(anchors):
            word = a[12 + (w + i * 7 % 32) % 32]
            r = i % 32
            word = ((word << r) | (word >> (32 - r))) & 0xffffffff if r else word
            bound.append(word ^ mix((0x71a5b39d ^ i * 0x9e3779b9 ^ w * 0x85ebca6b) & 0xffffffff))
        tie = mix((0xe17a1465 ^ w * 0xc2b2ae35) & 0xffffffff)
        value = 0
        for bit in range(32):
            ones = sum((word >> bit) & 1 for word in bound)
            value |= int(2 * ones > len(bound) or (2 * ones == len(bound) and (tie >> bit) & 1)) << bit
        vector.append(value)
    return tuple(vector)


def parse_windows(path, anchor_path):
    raw, old = path.read_bytes(), anchor_path.read_bytes()
    assert raw[:8] == b"HDBWIN03" and old[:8] == b"HDBANC02"
    h, a = struct.unpack_from("<14I", raw, 8), struct.unpack_from("<14I", old, 8)
    assert h[:2] == (3, 64) and h[6:9] == (176, 24, 64) and h[10:12] == (3, 24) and h[13] == 0
    anchors = [struct.unpack_from("<44I", old, 64 + i * 176) for i in range(a[4])]
    scoped = [anchor for anchor in anchors if anchor[1] == h[3]]
    assert len(scoped) == h[4]
    for i, anchor in enumerate(scoped):
        assert struct.unpack_from("<6I", raw, h[8] + i * 24) == (anchor[11], anchor[8], anchor[4], anchor[1], anchor[9], anchor[10])
    output = {}
    for layer in range(3):
        width, count, stride, offset, size, reserved = struct.unpack_from("<6I", raw, h[9] + layer * 24)
        assert width in (8, 16, 32) and stride == width // 2 and size == count * 176 and not reserved
        windows = [struct.unpack_from("<44I", raw, offset + i * 176) for i in range(count)]
        expected = sorted(set(range(0, len(scoped), stride)) | ({len(scoped) - width} if len(scoped) >= width else set())) if scoped else []
        if 0 < len(scoped) < width:
            expected = [0]
        assert [w[34] for w in windows] == expected
        for i, w in enumerate(windows):
            history, recorded_width, begin, length, first, last, first_anchor, last_anchor, obj, first_ver, last_ver, zero = w[32:]
            assert history == h[2] and recorded_width == width and length == min(width, len(scoped) - begin) and not zero
            start, end = scoped[begin], scoped[begin + length - 1]
            assert (first, last, first_anchor, last_anchor, obj, first_ver, last_ver) == (start[4], end[4], start[11], end[11], start[1], start[9], end[9])
            if i in {0, count // 2, count - 1} or length < width:
                assert w[:32] == fingerprint(scoped[begin:begin + length])
        output[width] = windows
    return output


def slots(windows, begin, count, budget):
    if not count:
        return [0xffffffff] * budget
    eligible = [begin + i for i in range(count) if windows[begin + i][35] == windows[begin + i][33]]
    eligible = eligible or list(range(begin, begin + count))
    return [eligible[s * (len(eligible) - 1) // (budget - 1)] for s in range(budget)]


def validate(output, anchors, roles_path):
    roles = {int(r["history_id"]): int(r["role"]) for r in rows(roles_path)}
    histories = {h: parse_windows(output / f"windows_{h}.bin", anchors / f"anchors_{h}.bin") for h in roles}
    score_rows = rows(output / "scores.csv")
    rank_rows, raw_top, query_rows = rows(output / "ranks.csv"), rows(output / "window_topk.csv"), rows(output / "queries.csv")
    for width in (8, 16, 32):
        windows, begins, counts = [], {}, {}
        for h in roles:
            begins[h], counts[h] = len(windows), len(histories[h][width])
            windows.extend(histories[h][width])
        raw = (output / f"similarities_{width}.bin").read_bytes()
        assert raw[:8] == b"HDBSIM03"
        header = struct.unpack_from("<6I", raw, 8)
        qrows = sorted((r for r in query_rows if int(r["width"]) == width), key=lambda r: int(r["query_row"]))
        assert header == (3, 32, width, len(qrows), len(windows), 2) and len(raw) == 32 + 2 * header[3] * header[4]
        matrix = array.array("H"); matrix.frombytes(raw[32:])
        if sys.byteorder != "little": matrix.byteswap()
        for row, qr in enumerate(qrows):
            query, index = int(qr["query_id"]), int(qr["window_id"])
            candidates = [i for i, w in enumerate(windows) if not roles[w[32]]]
            expected = sorted(candidates, key=lambda i: (matrix[row * len(windows) + i], i))[:5]
            observed = sorted((int(r["k"]), int(r["window_id"])) for r in raw_top if int(r["width"]) == width and int(r["query_row"]) == row)
            assert [i for _, i in observed] == expected
            checks = set(expected + ([0, len(windows) // 2, len(windows) - 1] if windows else []))
            if len(matrix) <= 10000: checks.update(range(len(windows)))
            for c in checks:
                distance = 1024 if index == 0xffffffff else sum((a ^ b).bit_count() for a, b in zip(windows[index][:32], windows[c][:32]))
                assert matrix[row * len(windows) + c] == distance
        queries = [q for q in roles if roles[q]]
        for group, q in enumerate(queries):
            for candidate in roles:
                if roles[candidate]: continue
                support = slots(windows, begins[candidate], counts[candidate], 8)
                d = [[1024 if c == 0xffffffff or int(qrows[group * 4 + r]["window_id"]) == 0xffffffff else matrix[(group * 4 + r) * len(windows) + c]
                      for c in support] for r in range(4)]
                values = {"best": min(map(min, d)) * 1024,
                          "normalized_top2": sum(sum(sorted(row)[:2]) for row in d) * 128,
                          "ordered": min(d[0][0] + d[1][a] + d[2][b] + d[3][7] for a, b in itertools.combinations(range(1, 7), 2)) * 256}
                for strategy, expected in values.items():
                    found = [r for r in score_rows if r["method"] == f"w{width}_{strategy}" and int(r["query_id"]) == q and int(r["candidate_id"]) == candidate]
                    assert len(found) == 1 and int(found[0]["score"]) == expected
    for method in {r["method"] for r in score_rows}:
        for q in roles:
            if not roles[q]: continue
            candidates = sorted((int(r["score"]), int(r["candidate_id"])) for r in score_rows if r["method"] == method and int(r["query_id"]) == q)
            ranked = sorted((int(r["k"]), int(r["score"]), int(r["candidate_id"])) for r in rank_rows if r["method"] == method and int(r["query_id"]) == q)
            assert [(d, c) for _, d, c in ranked] == candidates


def execute(cpu, gpu, v2cpu, directory, histories):
    directory.mkdir(parents=True, exist_ok=True)
    data, roles, anchors = directory / "events.bin", directory / "roles.csv", directory / "anchors"
    generate.v2.write_dataset(data, histories)
    with roles.open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(("history_id", "role")); writer.writerows((i, int(i % 3 == 2)) for i in range(len(histories)))
    run([v2cpu, data, "--roles", roles, "--out", anchors])
    for label, exe in (("cpu", cpu), ("gpu", gpu)):
        if exe:
            command = [exe, data, "--roles", roles, "--anchors", anchors, "--out", directory / label]
            if label == "gpu": command += ["--repeats", 1]
            run(command); validate(directory / label, anchors, roles)
    if gpu: parity(directory / "cpu", directory / "gpu")
    # Outcome mutations must not alter windows, scores, graph checks or ranks.
    raw = bytearray(data.read_bytes()); header = struct.unpack_from("<14I", raw, 8)
    for i in range(header[2]):
        label = struct.unpack_from("<I", raw, header[11] + 16 * i + 4)[0]
        struct.pack_into("<3I", raw, header[11] + 16 * i + 4, 1 - label, 99, 77)
    data.write_bytes(raw)
    run([cpu, data, "--roles", roles, "--anchors", anchors, "--out", directory / "changed"])
    parity(directory / "cpu", directory / "changed")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--cpu", type=Path, required=True); p.add_argument("--gpu", type=Path)
    p.add_argument("--v2-cpu", type=Path, required=True); args = p.parse_args()
    cpu, gpu, v2cpu = args.cpu.resolve(), args.gpu.resolve() if args.gpu else None, args.v2_cpu.resolve()
    Path("target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v3_test_", dir="target") as temporary:
        root = Path(temporary)
        histories = [generate.v2.history(n, 0, i % 2, i // 2, "dense_target", 991) for i, n in enumerate((0, 1, 7, 8, 9, 16, 17, 31, 32, 33, 257, 1025, 32768))]
        execute(cpu, gpu, v2cpu, root / "boundaries", histories)
        far = [generate.far_history(4096, 256, f, o, r, 991) for f in range(2) for o in (0, 1) for r in range(3)]
        execute(cpu, gpu, v2cpu, root / "far", far)
        catalog = {i: (f, o) for i, (f, o, r) in enumerate((f, o, r) for f in range(2) for o in (0, 1) for r in range(3))}
        for row in rows(root / "far/cpu/graph_checks.csv"):
            assert int(row["match"]) == (catalog[int(row["query_id"])] == catalog[int(row["candidate_id"])])
        assert any(int(r["object"]) == 2 and int(r["query_consumer_step"]) - int(r["query_producer_step"]) > 256 and
                   r["query_consumer_window"] != r["query_producer_window"] for r in rows(root / "far/cpu/dependency_links.csv"))
    print("PASS: v3 binary/partial windows, every GPU matrix/aggregate/rank, independent sampled Hamming, far provenance, and no outcome leakage")


if __name__ == "__main__": main()
