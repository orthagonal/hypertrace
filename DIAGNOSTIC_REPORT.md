# HDC diagnostic report

Evaluated the existing 9,000-record `data/safety_v0.bin`, with the original 6,000-record
training split and unchanged 1,024-bit CUDA prototype learner. Dataset SHA-256:
`e9f6489f6c46ac1d0b8971330346241a7881961424b069bbd56e6c8a3009c25b`.
GPU: RTX 2060 SUPER. CPU and GPU diagnostic artifacts match exactly. All experiments
below retain the learned prototypes; labels are used only to score predictions.

The evidence favors recognition of persistent state over useful temporal relationships.
Removing all recent actions improves held-out accuracy; recent actions alone are near
chance. Order perturbations do not consistently hurt. This is evidence about this
fitted model on this dataset, not proof that HDC cannot learn temporal structure.

## Accuracy by observed length

| Split | Actions | Records | HDC | Rules |
|---|---:|---:|---:|---:|
| Validation | 10–16 | 129 | 77.52% | 93.02% |
| Validation | 17–32 | 297 | 80.13% | 89.56% |
| Validation | 33–64 | 574 | 81.71% | 93.90% |
| Test | 10–16 | 118 | 81.36% | 93.22% |
| Test | 17–32 | 288 | 73.26% | 89.58% |
| Test | 33–64 | 594 | 79.80% | 91.75% |
| Long-test | 128–256 | 336 | 79.76% | 92.86% |
| Long-test | 257–512 | 664 | 80.57% | 91.57% |

There is no consistent accuracy decline with length. Length is not encoded. Persistent
flags retain exact state, while only the eight newest actions enter the HDC input.

## Observed cohorts and outcome classes

Actual scenario types and violation subcategories are not stored. The following are
disjoint **observed-prefix cohorts**, not reconstructed scenario metadata.

| Test cohort | Records | HDC | Rules |
|---|---:|---:|---:|
| Public payload | 118 | 87.29% | 100.00% |
| Local stage | 118 | 94.92% | 100.00% |
| Approval observed | 85 | 30.59% | 100.00% |
| Trusted inspection observed | 186 | 72.04% | 74.73% |
| Sensitive, untrusted, external, unapproved | 493 | 82.35% | 91.89% |

HDC produces 59 false positives among the 85 observed-approved records: a substantial
failure to use a directly available exception. Across all test records it recognizes
75.4% of allowed outcomes and 80.8% of attempted violations, versus 92.0% and 90.6%
for rules. There is only one generated violation family: unauthorized external transfer
of sensitive-derived content without approval.

## Prototype geometry and disagreements

Prototypes differ in only 66 of 1,024 bits. For the test split:

| HDC result | Records | Mean SAFE distance | Mean UNSAFE distance | Mean margin |
|---|---:|---:|---:|---:|
| True positive | 404 | 215.00 | 203.24 | +11.76 |
| True negative | 377 | 241.40 | 254.31 | −12.91 |
| False positive | 123 | 238.89 | 230.33 | +8.57 |
| False negative | 96 | 221.24 | 229.39 | −8.15 |

Margin is SAFE distance minus UNSAFE distance; positive predicts UNSAFE. There are
41 ties, all predicted SAFE, including 17 missed violations. Margins are bit differences,
not confidence probabilities. Every record's two distances and margin are exported.

HDC corrects four rule errors; rules correct 136 HDC errors. All 140 disagreements
are exported with observations. Illustrative zero-based file indices:

| Index | Oracle | HDC / rules | SAFE / UNSAFE distance | Margin |
|---|---|---|---|---:|
| 7219 | Allowed | SAFE / UNSAFE | 202 / 204 | −2 |
| 7360 | Attempted violation | UNSAFE / SAFE | 282 / 276 | +6 |
| 7005 | Allowed, approval observed | UNSAFE / SAFE | 239 / 235 | +4 |
| 7009 | Allowed, local stage | UNSAFE / SAFE | 263 / 261 | +2 |

Three of HDC's four corrections are allowed records; only one is a violation that rules
missed. Thus OR adds 86 false positives to catch one additional violation, as before.

## Feature and order interventions

| Intervention | Validation accuracy | Test accuracy | Long-test accuracy |
|---|---:|---:|---:|
| Original | 80.70% | 78.10% | 80.30% |
| State flags only | 83.70% | 83.40% | 84.30% |
| Recent actions only | 51.70% | 49.00% | 50.70% |
| Reverse recent actions | 79.80% | 79.30% | 80.30% |
| Shuffle recent actions, seed 42 | 79.70% | 81.10% | 81.00% |

Largest test losses from individual token removal: payload sensitivity −10.4 percentage
points, external stage −6.0, untrusted inspection −4.9, local stage −4.7. Removing
approval costs only 1.7 points. Individual recent-position removals change accuracy
by −0.4 to +1.0 points. All twenty feature results, including FP/FN and prediction
flips, are in `ablations.csv`.

The model responds to security-relevant state, but the approval failure and gains after
removing recent actions suggest incomplete exception handling and distracting input.
Removing unrelated-private-read and web-search tokens also costs 4.6 and 4.3 points.
Those losses cannot establish causal importance: omission changes the voting threshold
and moves examples away from the distribution used to learn the fixed prototypes.

## Limits and policy interpretation

- No full histories, original scenario IDs, true final destinations, or late-approval
  markers are stored. Noisy inspection observations and approvals arriving after the
  checkpoint create uncertainty the model cannot resolve from these records.
- Long-test's last eight actions are filler operations under the current generator;
  long-history performance therefore does not demonstrate learned temporal reasoning.
- These are single-seed results on one synthetic grammar. Cohorts are small, correlated
  flags are redundant, and many interventions were inspected without an independent
  confirmation set. Order sensitivity is not evidence of understanding causal order.
- The oracle label records whether a later violation is attempted. It is not an executed
  allow/block decision. Prediction FP/FN do not measure policy enforcement failures;
  enforcement outcomes are explicitly `not_recorded`. No live enforcer is exercised.

## Reproduction and validation

After building the executables as described in README:

```powershell
.\hdc_safety_gpu.exe data/safety_v0.bin --verify --diagnostics target/diagnostics_gpu
.\hdc_reference.exe data/safety_v0.bin --diagnostics target/diagnostics_cpu
```

Artifacts: `summary.txt`, `records.csv`, `disagreements.csv`, `groups.csv`, `ablations.csv`.
Validation: all three Rust tests and existing reporting tests pass; new diagnostic unit
and integration tests pass. GPU verification checks encodings, prototypes, every prediction,
and margins. The integration test exercises empty splits, length boundaries, no metadata
leakage, unchanged original output, and CPU/GPU artifact equality. Generator, dataset,
production encoder, and CUDA kernels/training are unchanged. No commit was created.
