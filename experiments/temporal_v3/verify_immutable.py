"""Hash all previous experiments and original code/data around the v3 workflow."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("record", "verify")); p.add_argument("snapshot", type=Path)
    args = p.parse_args()
    if args.mode == "record":
        paths = []
        for root in ("experiments/temporal_v1", "experiments/temporal_v2", "src", "include", "cuda", "tests"):
            paths.extend(p for p in Path(root).glob("*") if p.is_file())
        paths.extend(Path(p) for p in ("scripts/run_windows.ps1", "scripts/run_temporal_v1.ps1", "scripts/run_temporal_v2.ps1", "data/safety_v0.bin"))
        args.snapshot.write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}, indent=2))
    else:
        for name, digest in json.loads(args.snapshot.read_text()).items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        print("PASS: v0/v1/v2 source, tests, classifiers and original data unchanged")


if __name__ == "__main__": main()
