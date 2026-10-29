# Temporal Memory v2: Sparse Anchors

An additional ordered index beside the unchanged v1 tree. No classifier, trained
prototype changes, service or application framework is introduced.

- [FORMAT.md](FORMAT.md): significance, exact versions/provenance, binary layout,
  positional encoding, exact retrieval baseline and metric definitions.
- [REPORT.md](REPORT.md): measured v1/global/scoped retrieval, distances, memory and timings.

## Run

From a CUDA/MSVC developer PowerShell in the repository root:

```powershell
.\scripts\run_temporal_v2.ps1
```

Requires Rust, Python 3, CUDA and a C++17 host compiler. `-Cxx` may select a configured
g++ compiler, while nvcc still requires MSVC. The default CUDA architecture is sm_75
(RTX 2060 SUPER), matching v1. `-Repeats` defaults to 15. `-OutputDirectory` defaults
to `target/temporal_v2`; generated datasets and binary archives stay under that directory.

The script runs the unchanged v1 workflow (including all Rust/v1 tests), original
reporting/diagnostic tests and GPU verification, then builds/tests v2, generates twelve
collections, checks CPU/GPU artifacts, and creates a report. Existing v0/v1 source/data
hashes are checked before and after. The original classifier executable is not replaced.

## Individual experiment commands

After configuring the compiler environment:

```powershell
cl /nologo /O2 /EHsc /std:c++17 /Fetarget/temporal_v2_cpu.exe /Fotarget/temporal_v2_cpu.obj experiments/temporal_v2/cpu.cpp
nvcc -O3 -std=c++17 -arch=sm_75 experiments/temporal_v2/gpu.cu -o target/temporal_v2_gpu.exe
python -B experiments/temporal_v2/generate.py --out target/temporal_v2/data
python -B experiments/temporal_v2/run_collections.py --data target/temporal_v2/data --out target/temporal_v2/results --cpu target/temporal_v2_cpu.exe --gpu target/temporal_v2_gpu.exe --repeats 15
python -B experiments/temporal_v2/make_report.py --data target/temporal_v2/data --results target/temporal_v2/results --out target/temporal_v2/REPORT.md
```

The standalone corpus generator accepts `--seed`, `--families` (1–4), `--replicas`
(last is query; at least two), and `--lengths` (default 32,1024,4096,32768). It writes
the existing versioned HDBTMP01 format; the Rust v0/v1 generators remain untouched.
Each executable accepts one dataset, `--roles`, `--out`, `--object` (default 2), and
`--repeats`. The object is an explicit query constraint, never inferred from an oracle.

## Outputs

Per collection and CPU/GPU implementation:

- `anchors_<id>.bin`: every selected exact event, writer/source versions, original
  step, original full-table provenance references, HDC vector, fingerprints and exact key.
- `v1_history_0.mem`: one unchanged full v1 tree archive; every full tree is built,
  verified and resident separately during sparse construction.
- `fingerprints.csv`, `patterns.csv`: input-derived representations, without labels.
- `distances.csv`, `rankings.csv`: held-out query/gallery scores and top-5 results.
- `memory.csv`, `timings.csv`, `correctness.csv`: resource use, kernel samples and parity checks.

Aggregate outputs include `query_metrics.csv` (recall@1/2/5, false matches, ties),
`paired_distances.csv`, `resources.csv`, input hashes, and the generated report.
Top-K is checked on CUDA and independently on CPU; Python recomputes scores and ranks.

## Validation and interpretation

```powershell
cl /nologo /O2 /EHsc /std:c++17 /Fetarget/temporal_v2_tests.exe /Fotarget/temporal_v2_tests.obj experiments/temporal_v2/tests.cpp
.\target\temporal_v2_tests.exe
python -B experiments/temporal_v2/test_metrics.py
python -B experiments/temporal_v2/test_integration.py --cpu target/temporal_v2_cpu.exe --gpu target/temporal_v2_gpu.exe
```

Omit `--gpu` for CPU-only integration checks. Tests cover exact significance (including
significant filler), stable compaction boundaries, dense 32,768-event histories,
version reconstruction, non-anchor producers, vector equations, ordering, gap invariance,
empty streams, unknown versions, ties, candidate eligibility, CPU/GPU byte parity,
future-event rejection, and outcome-metadata mutation with unchanged fingerprints/rankings.

Object scoping gives a fair explicit-object query, but does not solve automatic relevance
discovery. Flat majority can still dilute changes among many anchors on the same object.
The controlled four-family task does not establish arbitrary provenance-graph or unseen
tool semantics. Distinct fingerprints alone are not evidence of useful retrieval; inspect
held-out recall and false-match results, especially the dense-target stress case.
