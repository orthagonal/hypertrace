"""Independently search every stored window for an arbitrary indexed query window."""
import argparse
import csv
from pathlib import Path
import struct
import sys


def read(path, width):
    data = path.read_bytes()
    assert data[:8] == b"HDBWIN03"
    h = struct.unpack_from("<14I", data, 8)
    assert h[:2] == (3, 64)
    for layer in range(h[10]):
        w, count, stride, offset, size, reserved = struct.unpack_from("<6I", data, h[9] + layer * 24)
        if w == width:
            return [struct.unpack_from("<44I", data, offset + i * 176) for i in range(count)]
    raise ValueError("width absent")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, required=True); p.add_argument("--history", type=int, required=True)
    p.add_argument("--width", type=int, choices=(8, 16, 32), required=True); p.add_argument("--window", type=int, required=True)
    p.add_argument("--top-k", type=int, default=5); args = p.parse_args()
    available = read(args.index / f"windows_{args.history}.bin", args.width)
    if args.window < 0 or args.window >= len(available) or args.top_k < 1:
        p.error("window must be an existing nonnegative index and top-k must be positive")
    query = available[args.window]
    scores = []
    for file in args.index.glob("windows_*.bin"):
        for local, candidate in enumerate(read(file, args.width)):
            d = sum((a ^ b).bit_count() for a, b in zip(query[:32], candidate[:32]))
            scores.append((d, candidate[32], local, candidate))
    scores.sort(key=lambda r: r[:3])
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("rank", "history_id", "window", "width", "first_step", "last_step", "object", "first_version", "last_version", "distance"))
    for rank, (distance, history, local, w) in enumerate(scores[:args.top_k], 1):
        writer.writerow((rank, history, local, args.width, w[36], w[37], w[40], w[41], w[42], distance))


if __name__ == "__main__": main()
