"""Record/check existing v0/v1 code/data hashes around a v2 experiment run."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("record", "verify"))
    p.add_argument("snapshot", type=Path)
    args = p.parse_args()
    if args.mode == "record":
        paths = [*Path("experiments/temporal_v1").glob("*"), *Path("cuda").glob("*"),
                 *Path("include").glob("*"), *Path("tests").glob("*"), Path("src/main.rs"), Path("src/temporal_v1.rs"),
                 Path("scripts/run_temporal_v1.ps1"), Path("scripts/run_windows.ps1"), Path("data/safety_v0.bin")]
        values = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}
        args.snapshot.write_text(json.dumps(values, indent=2))
    else:
        for name, digest in json.loads(args.snapshot.read_text()).items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        print("PASS: existing v0/v1 source, full-tree kernels and original data remain byte-identical")


if __name__ == "__main__":
    main()
