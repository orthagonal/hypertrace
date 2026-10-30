"""Run/compare CPU and CUDA with an enforced validation-before-test lock."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from selection import validate_freeze


def run(command):
    result = subprocess.run([str(c) for c in command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    print(result.stdout.strip(), flush=True)


def parity(cpu, gpu):
    for file in cpu.iterdir():
        if file.name not in ("summary.txt", "timings.csv", "graph_timings.csv"):
            assert file.read_bytes() == (gpu / file.name).read_bytes(), file.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--anchors", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--frozen", type=Path)
    args = p.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    if any(item["phase"] == "test" for item in manifest):
        assert args.frozen, "test retrieval requires a validation freeze"
        validate_freeze(args.frozen)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "evaluation_lock.json").write_text(json.dumps({"frozen_sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
            "phase": "test", "settings_file": str(args.frozen.resolve())}, indent=2))
    for item in manifest:
        source = args.data / item["name"]
        for label, exe in (("cpu", args.cpu.resolve()), ("gpu", args.gpu.resolve())):
            command = [exe, source / "events.bin", "--roles", source / "roles.csv", "--anchors", args.anchors / item["name"] / "gpu",
                       "--out", args.out / item["name"] / label]
            if label == "gpu":
                command += ["--repeats", args.repeats]
            run(command)
        parity(args.out / item["name"] / "cpu", args.out / item["name"] / "gpu")
        print(f"PASS v3 byte parity: {item['name']}", flush=True)


if __name__ == "__main__":
    main()
