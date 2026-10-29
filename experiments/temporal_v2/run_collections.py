"""Run both implementations and verify deterministic artifacts for every corpus."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from test_integration import run, parity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=15)
    args = p.parse_args()
    for item in json.loads((args.data / "manifest.json").read_text()):
        source, output = args.data / item["name"], args.out / item["name"]
        for label, exe in (("cpu", args.cpu.resolve()), ("gpu", args.gpu.resolve())):
            command = [exe, source / "events.bin", "--roles", source / "roles.csv", "--out", output / label]
            if label == "gpu":
                command += ["--repeats", str(args.repeats)]
            print(run(command).strip(), flush=True)
        parity(output / "cpu", output / "gpu")
        print(f"PASS byte parity: {item['name']}", flush=True)


if __name__ == "__main__":
    main()
