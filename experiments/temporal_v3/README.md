# Temporal Memory v3: Windowed and Dependency-Aware Retrieval

An isolated index/query experiment beside unchanged v0/v1/v2 code and trained models.
It stores all overlapping 8/16/32-anchor windows and compares held-out retrieval with
v1 roots, v2 global/scoped fingerprints, and the older exact baseline.

- [FORMAT.md](FORMAT.md): exact layout, scoring/normalization, query policy, dependencies and freeze rules.
- [REPORT.md](REPORT.md): measured held-out comparison, errors, ties, resources and timing.
- [frozen_settings.json](frozen_settings.json): the measured validation choice and code/data hashes.

## One command

From a CUDA/MSVC developer PowerShell at the repository root:

```powershell
.\scripts\run_temporal_v3.ps1
```

Requirements match v2: Rust, Python 3, a C++17 compiler, CUDA, and sm_75 for the RTX
2060 SUPER. `-Cxx`, `-Repeats` (default 15) and `-OutputDirectory` (default
`target/temporal_v3`) are configurable. The script runs all unchanged v0/v1/v2 tests,
builds/tests v3, evaluates validation cases, freezes settings/code hashes, then creates
and evaluates held-out cases. Original source/data hashes are checked afterward.

The reference and CUDA executables accept:

```powershell
temporal_v3_gpu.exe events.bin --roles roles.csv --anchors preserved_v2_gpu_directory --out output --repeats 15
temporal_v3_cpu.exe events.bin --roles roles.csv --anchors preserved_v2_gpu_directory --out output
```

`--object` defaults to 2, the explicit object query used by v2. Inputs are the unchanged
observed dataset and v2 archives. V3 verifies anchors, versions, vectors and baselines
against the original scalar algorithms before indexing them. No scorer catalog is
opened by either executable. No oracle key enters window scoring or graph verification.

## Retrieval choices

Every window is stored and compared during full-gallery window search. History scoring
uses a fixed budget of four query contexts and eight candidate contexts. This avoids
an ever-growing minimum over longer histories, but is an explicit bounded-context
query policy, not a claim of exhaustive arbitrary-motif history ranking.

Validation compares all three widths with best-window, normalized top-two means and
ordered multi-window matching. Its frozen winner is used for the primary test report.
Other fixed test settings remain ablations, not a source of tuning.

The optional CPU dependency stage verifies proposed candidates by walking raw typed
dependency DAGs rooted at the first/last observed scoped events. It preserves exact
version aliases and chronology and may abstain. This is not a precomputed answer
signature or arbitrary-graph matcher. Candidate budget and candidate recall are
reported; checking all 16 gallery histories is explicitly identified if selected.

## Artifacts

- `frozen_settings.json`: validation choice, all validation scores, source/data hashes.
- `test/results/evaluation_lock.json`: freeze hash checked before held-out retrieval.
- `windows_<id>.bin`: all widths, exact member references, steps and writer versions.
- `similarities_<width>.bin`: every query-context-to-window Hamming distance.
- `window_topk.csv`, `scores.csv`, `ranks.csv`: full-window search and history aggregation.
- `graph_checks.csv`, `dependency_links.csv`, `dependency_ranks.csv`: exact verification after proposals.
- `memory.csv`, `timings.csv`, `graph_timings.csv`, `correctness.csv`: resource/verification evidence.
- Aggregated `query_metrics.csv`, `resources.csv`, `selected_dependency_links.csv`, and report.
- `test_input_sha256.json`: binds observed histories, role files and preserved v2 archives.

Original histories and v1/v2 archives remain available. References may point outside
the scoped index or to non-anchor producers; exact provenance is never reconstructed
from HDC similarity. All artifacts stay in the selected output directory.

Any interior window can also be searched independently of the benchmark context policy:

```powershell
python -B experiments/temporal_v3/search_windows.py --index output --history 0 --width 8 --window 12 --top-k 5
```

That utility independently reads the binary index and computes scalar Hamming scores;
CUDA's arbitrary-index comparison/top-K kernels provide the corresponding GPU primitive.

## Tests

```powershell
temporal_v3_tests.exe
python -B experiments/temporal_v3/test_metrics.py
python -B experiments/temporal_v3/test_integration.py --cpu temporal_v3_cpu.exe --gpu temporal_v3_gpu.exe --v2-cpu temporal_v2_cpu.exe
```

The C++ tests cover overlap/tails, no truncation, order sensitivity, count normalization,
exact source/version aliases, serialization-independent edge ordering, and a 32,768-node
dependency chain. Integration verifies every GPU window, score, aggregate and rank against
the independent scalar implementation. Python independently checks binary fields, vector
samples, ranking equations and distant links. Outcome mutation must leave all retrieval
artifacts unchanged. A regression has identical old answer keys but different version
aliases and is rejected by raw dependency verification.

No HTTP service, application framework, classifier or automatic commit is introduced.
