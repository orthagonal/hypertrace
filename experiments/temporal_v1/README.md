# Temporal Memory v1

An isolated, reproducible 32-way GPU event-memory experiment. It creates no classifier
and does not read or change the existing learned prototypes. Existing CUDA kernels,
timings and v0 data generation remain unchanged.

Read [FORMAT.md](FORMAT.md) for the complete binary/encoding contract and
[REPORT.md](REPORT.md) for the measured results, including a long-history root collision.

## One command (Windows)

From a CUDA/MSVC developer PowerShell, in the repository root:

```powershell
.\scripts\run_temporal_v1.ps1
```

The script requires Rust, Python 3 and a C++17 host compiler. `-Cxx` can select a
configured g++ compiler. `nvcc` still requires its MSVC host environment. The default
architecture is `sm_75` for RTX 2060 SUPER, matching the existing lab environment.
Outputs go to `target/temporal_v1`; `-OutputDirectory` and `-Repeats` are configurable.
It builds, tests and runs both implementations, compares deterministic artifacts,
and creates a report. It does not execute or overwrite the existing classifier binary.

## Individual commands

```powershell
cargo run --release -- --temporal-v1 --out target/temporal_v1/events.bin
g++ -O3 -std=c++17 experiments/temporal_v1/cpu.cpp -o target/temporal_cpu.exe
nvcc -O3 -std=c++17 -arch=sm_75 experiments/temporal_v1/gpu.cu -o target/temporal_gpu.exe
.\target\temporal_cpu.exe target/temporal_v1/events.bin --out target/temporal_v1/cpu
.\target\temporal_gpu.exe target/temporal_v1/events.bin --out target/temporal_v1/gpu --repeats 15
python experiments/temporal_v1/make_report.py --dataset target/temporal_v1/events.bin --results target/temporal_v1/gpu --cpu-results target/temporal_v1/cpu --out target/temporal_v1/REPORT.md
```

Rust generation also accepts `--seed`, `--temporal-pairs`, and `--temporal-lengths`
(comma-separated, default `32,1024,4096,32768`). Lengths 0–4 are edge cases rather
than policy scenarios. With the same seed/arguments, files are byte deterministic.
Every pair shares filler actions but exchanges approval and rewrite order. The
simulator tracks current-version approval exactly; the later transfer is not exported.
This v1 grammar deliberately adds object/version semantics that v0 did not store.

## Artifacts

- `history_<id>.mem`: exact events, dependency edges, vectors and all hierarchy levels.
- `memory.csv`: resident records/bytes at every level, including empty histories.
- `timings.csv`: CUDA event medians/min/max per kernel, level, variant and history.
- `correctness.csv`: every leaf/parent/mask/query/search verification result.
- `query_distances.csv`: similarity to the earliest observed private-read event (fixed
  fallback query when absent), for every leaf and internal summary.
- `retrieval.csv`, `retrieved_events.csv`, `regions.csv`: exhaustive property counts,
  exact event locations and matching internal regions for all eight properties.
- `sensitivity.csv`: root and per-level distances for policy order pairs and separate
  controlled reorder/removal/gap perturbations. Empty-root distances are `NA`.

CUDA uses separate kernels for event encoding, one-level construction, query comparison,
exact mask descent and leaf filtering. The host launches successive levels. GPU output
is **always** checked against an independent scalar CPU implementation. Every field of
every leaf and parent is compared, including exact masks. Exact queries never inspect
HDC vectors; dependency edges remain a separate exact side table, not mask semantics.

## Tests

```powershell
cargo test
g++ -O3 -std=c++17 experiments/temporal_v1/tests.cpp -o target/temporal_tests.exe
.\target\temporal_tests.exe
python experiments/temporal_v1/test_integration.py --generator target/release/hypercubedb-safety-sim.exe --cpu target/temporal_cpu.exe --gpu target/temporal_gpu.exe
```

Omit `--gpu` for CPU-only contract tests. The integration test generates independent
edge-case lengths, replays oracle outcomes in Python, checks serialized layouts and
exact queries, mutates scoring metadata to test leakage, rejects malformed/future
records, and regenerates v0 with seed 42 to check its original SHA-256. Unit tests
also replace summaries with arbitrary bits and confirm exact retrieval is unchanged.

No claim is made that lossy summaries preserve useful temporal information. In
particular, opposite-outcome long histories can have identical root summaries while
exact masks and raw events still preserve their complete observed event locations.
