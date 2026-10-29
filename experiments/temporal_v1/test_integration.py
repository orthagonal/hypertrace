"""Independent binary, exact-mask, leakage and regression checks. No third-party deps."""
import argparse
import csv
import hashlib
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

PROPERTIES = ("private_read", "external", "approval", "derived", "sensitive", "local", "untrusted", "filler")


def run(args, success=True):
    result = subprocess.run([str(a) for a in args], text=True, capture_output=True)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result.stdout


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def dataset(path):
    raw = path.read_bytes()
    assert raw[:8] == b"HDBTMP01"
    h = struct.unpack_from("<14I", raw, 8)
    assert h[:2] == (1, 64) and h[5:9] == (32, 12, 32, 64) and h[12:] == (16, 0)
    assert h[9] == 64 + h[2] * 32 and h[10] == h[9] + h[3] * 32
    assert h[11] == h[10] + h[4] * 12 and len(raw) == h[11] + h[2] * 16
    histories = []
    for i in range(h[2]):
        d = struct.unpack_from("<8I", raw, 64 + i * 32)
        events = [struct.unpack_from("<8I", raw, h[9] + (d[2] + j) * 32) for j in range(d[3])]
        edges = [struct.unpack_from("<3I", raw, h[10] + (d[4] + j) * 12) for j in range(d[5])]
        outcome = struct.unpack_from("<4I", raw, h[11] + i * 16)
        assert d[0] == outcome[0] == i and d[6] == len(events)
        assert all(e[4] < d[6] and e[0] != 18 for e in events)
        # Replay the current-version approval rule independently of the Rust oracle.
        sensitive, approved = {}, {}
        seen = {}
        for e in events:
            action, obj, source, flags, step, begin, count, reserved = e
            assert not reserved
            for edge in edges[begin:begin + count]:
                assert edge[0] == step and edge[1] < step and seen[edge[1]][1] == edge[2]
            if action in (1, 8):
                sensitive[obj], approved[obj] = True, False
            elif action == 4:
                sensitive[obj], approved[obj] = sensitive[source], False
            elif action == 11:
                approved[obj] = True
            elif action == 15:
                sensitive[obj], approved[obj] = False, False
            seen[step] = e
        expected = len(events) >= 5 and sensitive.get(2, False) and not approved.get(2, False)
        assert outcome[1] == expected
        histories.append((d, events, edges, outcome))
    for a, b in zip(histories[::2], histories[1::2]):
        assert Counter(e[:4] for e in a[1]) == Counter(e[:4] for e in b[1])
        if len(a[1]) >= 5:
            assert (a[3][1], b[3][1]) == (0, 1)
    return raw, h, histories


def read_memory(path, original):
    raw = path.read_bytes()
    assert raw[:8] == b"HDBMEM01"
    h = struct.unpack_from("<14I", raw, 8)
    assert h[:2] == (1, 64) and h[6:11] == (160, 176, 12, 24, 64)
    assert h[3] == len(original[1]) and h[4] == len(original[2])
    levels = []
    for l in range(h[5]):
        desc = struct.unpack_from("<6I", raw, 64 + l * 24)
        assert desc[0] == l and desc[2] == (176 if l else 160) and desc[4] == desc[1] * desc[2]
        fmt = "<44I" if l else "<40I"
        levels.append([struct.unpack_from(fmt, raw, desc[3] + j * desc[2]) for j in range(desc[1])])
    leaves = levels[0]
    assert [leaf[:8] for leaf in leaves] == original[1]
    edges = [struct.unpack_from("<3I", raw, h[12] + i * 12) for i in range(h[4])]
    assert edges == original[2]
    for l in range(1, len(levels)):
        for i, node in enumerate(levels[l]):
            start, child, count, properties = node[32:36]
            assert child == i * 32 and count == min(32, len(levels[l - 1]) - child)
            assert start == (levels[l - 1][child][4] if l == 1 else levels[l - 1][child][32])
            expected_union = 0
            for p in range(8):
                expected = 0
                for c in range(count):
                    flags = levels[l - 1][child + c][3 if l == 1 else 35]
                    if flags & (1 << p):
                        expected |= 1 << c
                assert node[36 + p] == expected
                if expected:
                    expected_union |= 1 << p
            assert properties == expected_union
    return levels


def validate(directory, histories):
    retrieval = csv_rows(directory / "retrieved_events.csv")
    regions = csv_rows(directory / "regions.csv")
    counts = csv_rows(directory / "retrieval.csv")
    distances = csv_rows(directory / "query_distances.csv")
    for i, original in enumerate(histories):
        levels = read_memory(directory / f"history_{i}.mem", original)
        for p, name in enumerate(PROPERTIES):
            expected = [j for j, e in enumerate(original[1]) if e[3] & (1 << p)]
            got = [int(r["event_index"]) for r in retrieval if int(r["history_id"]) == i and r["property"] == name]
            assert got == expected
            summary = [r for r in counts if int(r["history_id"]) == i and r["property"] == name]
            assert len(summary) == 1 and int(summary[0]["exact_matches"]) == len(expected) and summary[0]["verified"] == "PASS"
            expected_regions = {(l, j) for l in range(1, len(levels)) for j, n in enumerate(levels[l]) if n[36 + p]}
            got_regions = {(int(r["level"]), int(r["node_index"])) for r in regions if int(r["history_id"]) == i and r["property"] == name}
            assert expected_regions == got_regions
        # Query is the earliest private read. For empty/no-private histories only
        # range-check distances; the C++ reference verifies the fallback query.
        query = next((leaf[8:] for leaf in levels[0] if leaf[3] & 1), None)
        this_distances = [r for r in distances if int(r["history_id"]) == i]
        assert len(this_distances) == sum(map(len, levels))
        for row in this_distances:
            l, index = int(row["level"]), int(row["node_index"])
            vector = levels[l][index][:32] if l else levels[l][index][8:]
            assert 0 <= int(row["hamming_distance"]) <= 1024
            if query is not None:
                assert int(row["hamming_distance"]) == sum((a ^ b).bit_count() for a, b in zip(query, vector))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path)
    args = parser.parse_args()
    generator, cpu = args.generator.resolve(), args.cpu.resolve()
    gpu = args.gpu.resolve() if args.gpu else None
    Path("target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="temporal_test_", dir="target") as temporary:
        base = Path(temporary)
        generated = base / "events.bin"
        command = [generator, "--temporal-v1", "--temporal-lengths", "0,1,2,5,31,32,33,1024,1025,4096,32768", "--out", generated]
        run(command)
        raw, h, histories = dataset(generated)
        run(command)
        assert generated.read_bytes() == raw, "non-deterministic generation"
        run([cpu, generated, "--out", base / "cpu"])
        validate(base / "cpu", histories)
        if gpu:
            run([gpu, generated, "--out", base / "gpu", "--repeats", "1"])
            validate(base / "gpu", histories)
            for file in (base / "cpu").iterdir():
                if file.name not in ("timings.csv", "summary.txt"):
                    assert file.read_bytes() == (base / "gpu" / file.name).read_bytes(), file.name
        # Flip only scoring metadata. Memory bytes MUST remain identical.
        flipped = bytearray(raw)
        for i in range(h[2]):
            struct.pack_into("<3I", flipped, h[11] + i * 16 + 4, 1 - histories[i][3][1], 99, 77)
        changed = base / "outcomes_changed.bin"
        changed.write_bytes(flipped)
        run([cpu, changed, "--out", base / "changed"])
        for file in (base / "cpu").glob("*.mem"):
            assert file.read_bytes() == (base / "changed" / file.name).read_bytes(), "oracle leakage"
        if gpu:
            run([gpu, changed, "--out", base / "changed_gpu", "--repeats", "1"])
            for file in (base / "cpu").glob("*.mem"):
                assert file.read_bytes() == (base / "changed_gpu" / file.name).read_bytes(), "GPU oracle leakage"
        for name, offset, value in (
            ("version", 8, 99),
            ("offset", 8 + 9 * 4, 1),
            ("future_event", h[9] + 4 * 4, 1),  # First nonempty event: history 2, cutoff=1.
            ("future_dependency", h[10] + 4, 0xffffffff),
            ("wrong_dependency_object", h[10] + 8, 999999),
        ):
            corrupt = bytearray(raw)
            struct.pack_into("<I", corrupt, offset, value)
            path = base / f"invalid_{name}.bin"
            path.write_bytes(corrupt)
            run([cpu, path, "--out", base / "invalid"], success=False)
        truncated = base / "truncated.bin"
        truncated.write_bytes(raw[:-1])
        run([cpu, truncated, "--out", base / "invalid"], success=False)
        v0 = base / "v0.bin"
        run([generator, "--out", v0, "--seed", "42"])
        assert hashlib.sha256(v0.read_bytes()).hexdigest() == "e9f6489f6c46ac1d0b8971330346241a7881961424b069bbd56e6c8a3009c25b", "v0 byte regression"
        print("PASS: deterministic paired histories; independent oracle replay; binary layout; all masks/locations; empty/partial/32768 histories; no oracle leakage; rejected future events; v0 byte regression; CPU/GPU parity")


if __name__ == "__main__":
    main()
