"""Exercise metric denominators, abstentions, and non-input metadata isolation."""
import argparse
import csv
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from run import check_rows, read_rows


def test(exe, original):
    with tempfile.TemporaryDirectory(prefix="candidate-v31-") as temp:
        temp = Path(temp)
        data = temp / "data"
        shutil.copytree(original, data)
        out = temp / "first"
        subprocess.run([str(exe), "--original", str(data), "--out", str(out), "--repeats", "1"], check=True)
        first = check_rows(out)
        raw = bytearray((data / "events.bin").read_bytes())
        header = struct.unpack_from("<14I", raw, 8)
        for i in range(header[2]):
            # pair_id, oracle label, scenario and category are never model inputs.
            struct.pack_into("<I", raw, 64 + 32 * i + 4, 123456 + i)
            offset = header[11] + i * 16
            label = struct.unpack_from("<I", raw, offset + 4)[0]
            struct.pack_into("<3I", raw, offset + 4, 1 - label, 777, 999)
        (data / "events.bin").write_bytes(raw)
        catalog = read_rows(data / "catalog.csv")
        with (data / "catalog.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=catalog[0]); writer.writeheader()
            for row in catalog:
                row["family"] = str(1000 + 3 * int(row["family"]))
                row["oracle_label"] = str(1 - int(row["oracle_label"]))
                writer.writerow(row)
        other = temp / "mutated"
        subprocess.run([str(exe), "--original", str(data), "--out", str(other), "--repeats", "1"], check=True)
        second = check_rows(other)
        assert (out / "scores.csv").read_bytes() == (other / "scores.csv").read_bytes()
        assert (out / "candidates.csv").read_bytes() == (other / "candidates.csv").read_bytes()
        for a, b in zip(first, second):
            assert {k: v for k, v in a.items() if not k.endswith("_ms")} == {k: v for k, v in b.items() if not k.endswith("_ms")}
        for strategy in ("hdc_verified", "inverted_verified", "dependency_verified", "hybrid_verified"):
            for r in (x for x in first if x["strategy"] == strategy):
                assert int(r["final_hits"]) <= int(r["candidate_hits"])
                if int(r["rejected"]) == int(r["verifications"]):
                    assert int(r["final_hits"]) == 0 and r["abstention"] == "1"
    print("PASS: K=1 maximum 50%, independent candidate/final recall, rejected/abstention accounting, oracle/family/pair metadata isolation")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--exe", type=Path, required=True); p.add_argument("--original", type=Path, required=True)
    a = p.parse_args(); test(a.exe.resolve(), a.original.resolve())
