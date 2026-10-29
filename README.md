# HypercubeDB — Predictive Agent Safety Lab (v0.1)

A **research prototype**, not a security product. Uses a synthetic tool-call world, deterministic exact policy labels, and a small GPU HDC classifier to ask whether the agent's *observed history* predicts a later policy-violation attempt.

This is intentionally focused on **GPU algorithms**, not HTTP servers, LLM integration, or a general-purpose database.

## Fast start (Windows + NVIDIA CUDA)

Requirements: Rust stable (`cargo`), CUDA Toolkit (`nvcc`), and Visual Studio C++ Build Tools configured for nvcc. Set `sm_75` for an RTX 2060 SUPER.

From PowerShell, in the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1
```

Equivalent manual steps:

```powershell
cargo run --release -- --out data/safety_v0.bin --seed 42
nvcc -O3 -std=c++17 -arch=sm_75 -I include cuda/hdc_safety.cu -o hdc_safety_gpu.exe
.\hdc_safety_gpu.exe data/safety_v0.bin --verify
```

Optional CPU reference (MinGW g++):

```powershell
g++ -O3 -std=c++17 -I include cuda/hdc_reference.cpp -o hdc_reference.exe
.\hdc_reference.exe data/safety_v0.bin
```

You can also build the reference without Rust, using `python tests/make_fixture.py` and pointing `hdc_reference` at `data/fixture.bin` (format test, not real training data).

## Simulated behavior

- Reads private/public files and performs transformations (a derivation **retains** sensitivity).
- Inserts numerous unrelated reads, searches, edits, and saves; unrelated private reads don't taint the selected outbound payload.
- Inspects destinations, requests/receives approvals, and stages local or external transfers.
- Takes a **snapshot before** the final HTTP operation. Some authorizations can arrive after the snapshot.
- Final exact oracle: deny when sending sensitive-derived content to an unauthorized external destination without approval.
- Labels the snapshot `1` if a later forbidden transfer is attempted, otherwise `0`.

We alternate positive and negative scenarios. Negative controls vary one of four conditions: public payload, trusted destination, explicit approval, or local-only output. The model **cannot see** the final operation or any later approval.

### Splits

`train` 6,000 examples (10–64 observed actions); `validation` 1,000; `test` 1,000; `long-test` 1,000 (128–512 actions). All splits use independent trajectories from the same simulator grammar. **Long-test measures robustness to longer filler chains, not generalization to unseen tool semantics or real attacks.**

### CPU vs GPU responsibilities

**CPU (Rust):** generate auditable scenarios, maintain authoritative lineage/permissions, produce exact labels, write binary data. In future gateway: intercept and enforce tool access.

**GPU (CUDA):**

1. `encode_events`: 12 trusted Boolean feature states plus 8 recent action codes, each mapped to a deterministic pseudo-random hypervector; bitwise majority creates a 1,024-bit behavioral representation. Missing actions have a distinct padding token.
2. `learn_prototypes`: each of 1,024 GPU threads counts training-set bit votes for each class; emits two class prototypes. No backpropagation or large neural model.
3. `predict`: one warp per record compares its vector to both prototypes with XOR+popcount, sums 32 partial distances with warp shuffles, chooses the closer prototype.

Hard rule execution remains separate from HDC guesses. No GPU result can override an exact denial. The `basic_baseline` is a deliberately simple rule applied only to observed flags, not future facts.

## Binary data format

24-byte header: `HDBSAFE1`, little-endian `u32` version 1, number of records, vector dimensionality 1024, record word count 12. Each 48-byte record: `u32 split`, `u32 label`, `u32 flags`, `u32 prefix_len`, `u32 recent[8]`. See `include/safety_data.h` and `src/main.rs`.

These are *structured observations*, not raw text. Extraction of security-relevant facts from arbitrary model output is an unsolved integration challenge. In real tools, unknown data provenance must be treated conservatively.

## What this v0 does *not* prove

- Neither a security boundary for a live LLM nor guaranteed prompt-injection detection.
- No true infinite-memory or general long-horizon inference; `flags` hold exact persistent state, the HDC sketch only sees eight recent events plus current state.
- No training on real-world attacks, no measured transfer to unfamiliar agent tools or unobserved threat classes.
- This baseline's prototype model is deliberately simple and may underperform the rule-only baseline. That is valuable evidence.
- Runs are deterministic synthetic experiments, not proof that a production safety gateway will be safe.

## Validation

`cargo test` checks core provenance invariants and control labels. `hdc_safety_gpu.exe data/safety_v0.bin --verify` checks **all GPU encodings, learned prototypes, and predictions** against the CPU reference (it is slower, but appropriate for our first version). The CPU reference uses the same deterministic hypervector hash equations as CUDA, so it detects implementation divergence rather than independently proving the encoding is scientifically useful.

Kernel times are single executions, **not** stable benchmarks. Measure throughput separately after correctness is established.

Both executables append a paired prediction comparison after the existing per-model metrics.
For each split (including the in-sample training split), the four counts show both models
correct, both wrong, HDC correct with rules wrong, and rules correct with HDC wrong on
the same records. They sum to the split's record count. `HDC OR rules` flags a violation
if either prediction is positive and reports precision, recall, and false positives
alongside the existing accuracy and false-negative metrics. Precision is TP/(TP+FP)
and recall is TP/(TP+FN); an empty denominator reports zero, as in the existing metrics.
Use the held-out splits to assess whether HDC corrects rule errors and whether any
recall gain from combining them justifies the additional false positives.

The paired reporting unit test requires only a C++17 compiler:

```powershell
g++ -std=c++17 -I include tests/report_test.cpp -o report_test.exe
.\report_test.exe
```

## Diagnostic evaluation

Diagnostics are opt-in and append output after the existing metrics and paired
comparison. They run on the host using the learned prototypes; kernels, training,
and the dataset format stay unchanged. GPU verification also checks signed margins.

```powershell
.\hdc_safety_gpu.exe data/safety_v0.bin --verify --diagnostics target/diagnostics_gpu
.\hdc_reference.exe data/safety_v0.bin --diagnostics target/diagnostics_cpu
```

Each directory contains:

- `summary.txt`: length buckets, distances, margins, disagreements, and control experiments by split.
- `records.csv`: every record's zero-based file index, prefix observations, oracle label, predictions,
  SAFE/UNSAFE prototype distances, signed margin, and prediction-error categories.
- `disagreements.csv`: all examples where one model is correct and the other is wrong.
- `groups.csv`: both accuracies, counts, error counts and distance statistics by length,
  observed-state cohort, oracle outcome, and HDC prediction-error category.
- `ablations.csv`: all twelve Boolean facts and eight recent-action positions individually removed,
  plus state-only, recent-only, reversed-order and deterministic shuffled-order controls.
  It includes accuracy change in percentage points, FP/FN, flips, errors fixed and predictions broken.

Feature removal omits the entire token (including its zero state), recomputes the
majority denominator, and keeps the original prototypes. This measures sensitivity
of the fitted model, not the value of retraining without a feature. State-only and
recent-only use those same fixed prototypes. Order controls preserve the action
multiset; the shuffle has seed 42 and depends only on observed actions. The original
position convention is `recent[0]` newest through `recent[7]` oldest. The margin is
`distance(SAFE) - distance(UNSAFE)`; positive predicts UNSAFE, zero predicts SAFE.
Distances are Hamming bit counts, not calibrated probabilities.

No scenario type, violation subtype, full trajectory, final destination, late-approval
marker, or enforcement outcome is stored. Cohorts are explicitly descriptions of
observed flags, with priority public payload, local stage, approved, trusted inspection,
then sensitive/untrusted/external/unapproved, then other. They must not be treated as
true scenario labels. The oracle label is used only for retrospective scoring; FP/FN
describe predictions, not executed policy decisions. No live enforcer is tested.

Run the new tests after building both executables:

```powershell
g++ -O3 -std=c++17 -I include tests/diagnostics_test.cpp -o diagnostics_test.exe
.\diagnostics_test.exe
python tests/diagnostics_integration_test.py --cpu .\hdc_reference.exe --gpu .\hdc_safety_gpu.exe
```

The integration test also works with just `--cpu`. It checks artifacts, every length
boundary, empty splits, group accounting, absence of metadata leakage, original
output preservation, and CPU/GPU parity. See [DIAGNOSTIC_REPORT.md](DIAGNOSTIC_REPORT.md)
for the research dataset results and their limitations.

## Learning questions

The separate [Temporal Memory v1 experiment](experiments/temporal_v1/README.md) adds
optional full-history generation, a GPU 32-way hierarchy, exact mask queries, independent
CPU validation and measured order/removal/gap sensitivity. It does not change this
classifier or its prototypes. See its [binary layout](experiments/temporal_v1/FORMAT.md)
and [measured report](experiments/temporal_v1/REPORT.md).

[Temporal Memory v2: Sparse Anchors](experiments/temporal_v2/README.md) adds a separate
ordered anchor index and held-out temporal-pattern retrieval experiments. It retains
the full v1 tree and compares global/scoped anchor fingerprints with v1 and an exact
chronological/provenance baseline, including dense significant-event cases.

- Does HDC predict outcomes better than simple rules on held-out synthetic trajectories?
- How does longer distraction affect a last-eight-actions encoder when exact state persists?
- What is the cost of GPU encoding, learning, and prediction separately? (`ncu -k regex:encode_events ...` after the basic build runs.)
- Can a 4-thread subgroup per memory outperform a full warp when we optimize later? Keep this as a **future** benchmark; v0 uses 32 threads per memory for clarity.

## File structure

```
Cargo.toml                  Rust simulator (no dependencies)
src/main.rs                 Exact scenario generator + labeled snapshots
include/safety_data.h      Cross-language record and HDC encoding contract
include/read_dataset.h     Checked binary file reader
include/report.h           Accuracy and control-baseline metrics
include/diagnostics.h      Host-only distance, cohort, ablation and order diagnostics
cuda/hdc_safety.cu          CUDA encoder, learner, predictor
cuda/hdc_reference.cpp      CPU mathematical reference
scripts/run_windows.ps1     One-command demo
tests/make_fixture.py       Standalone binary-format fixture generator
tests/report_test.cpp       Paired prediction and OR classifier count checks
tests/diagnostics_test.cpp  Diagnostic encoder and scoring unit tests
tests/diagnostics_integration_test.py  Artifact, metadata-isolation and CPU/GPU checks
DIAGNOSTIC_REPORT.md        Measured diagnostic results and research limitations
```

GPU execution and CPU/GPU verification were exercised on an RTX 2060 SUPER for the
diagnostic report. Repeat `--verify` after changes and investigate unexpected
divergences before treating accuracy numbers as reproducible.
