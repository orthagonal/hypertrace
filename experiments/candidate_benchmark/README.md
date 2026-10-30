# Temporal Memory v3.1 candidate benchmark

This isolated experiment measures candidate generation before the **unchanged v3
exact verifier**. No classifiers, prototypes, old experiments or frozen HDC
settings are changed. See [STRATEGIES.md](STRATEGIES.md) for algorithms and the
[measured report](results/REPORT.md) for results.
Recorded checks are in [results/tests.txt](results/tests.txt).

From the `hypertrace` repository in PowerShell, with the existing completed v3
data/results in `target/temporal_v3`:

```powershell
.\scripts\run_candidate_benchmark.ps1
```

The script locates MSVC using `vswhere`, builds for `sm_75`, runs all existing
v0/v1/v2/v3 regression suites plus the new tests, then executes **112 finite
collections**, printing progress after each one. It exits when complete. It does
not rerun the old full benchmark suites or overwrite their data. CUDA, MSVC,
Python and Rust must be available. If automatic MSVC discovery fails:

```powershell
.\scripts\run_candidate_benchmark.ps1 -VcVarsPath 'E:\Visual Studio\Insiders\VC\Auxiliary\Build\vcvars64.bat'
```

For a short run or a restart:

```powershell
.\scripts\run_candidate_benchmark.ps1 -Lengths '32' -Galleries '16,128' -Repeats 2
.\scripts\run_candidate_benchmark.ps1 -Resume
```

Checkpoints are reused only when code, configuration, input hashes, repeated-run
count and output hashes agree. A reduced run is not the full research result.
Resume assumes the same machine and CUDA/MSVC toolchain; use a fresh output
directory or omit `-Resume` when either changes.
Ctrl+C stops the process; completed checkpoints remain usable. If original v3
outputs are absent, run the original `scripts/run_temporal_v3.ps1` first.

The output report is `target/candidate_benchmark/report/REPORT.md`. Each phase,
suite, gallery size and profile/length has `queries.csv` (individual outcomes and
latencies), `candidates.csv` (rank/acceptance traces), `scores.csv` (all frozen HDC
scores), `construction.csv`, `histories.csv` and a hash-checked completion marker.
`tests.log` contains regression output. The report generator produces compact
CSVs covering every measured configuration. Checked-in results contain those
CSVs, hardware information, settings and source hashes; executables and large
per-query traces remain in ignored `target/`.

The recorded full run used an RTX 2060 SUPER (8 GiB), Intel Core i9-10850K,
CUDA 13.4.92 and MSVC tools 14.51.36231. The checked-in CSVs contain measured
timings, not estimates of future hardware performance.

The additional galleries are reproducible in-memory recipes implemented only in
this experiment. The full observed events and raw edges are retained in RAM;
no history is truncated, and the previous dataset generator is unchanged.

No command stages or commits files, and no v4 experiment runs.
