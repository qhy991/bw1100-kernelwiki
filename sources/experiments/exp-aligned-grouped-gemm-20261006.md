---
id: exp-aligned-grouped-gemm-20261006
title: Group-order choices after aligned vectorized lowering
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, gemm, tiling, paired-timing, profiling, negative-result]
confidence: experimental
date: '2026-10-06'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-aligned-grouped-gemm-20261006
artifacts:
- grouped_gemm_probe.py
- profile_grouped_gemm.py
- prepare.jsonl
- compiled
- measure.jsonl
- confirm.jsonl
- profile-fetch.csv
- profile-fetch-checks.jsonl
- profile-fetch-validation.json
- profile-l2.csv
- profile-l2-checks.jsonl
- profile-l2-validation.json
- measure-admission-terminal.json
- confirm-admission-terminal.json
- profile-fetch-admission-terminal.json
- profile-l2-admission-terminal.json
- analyze.py
- verify_evidence.py
- accepted-analysis.json
compiler: native vendor Triton 3.6.0; aligned ASTSource; stages2; no Cake change
source_commit: e090b412
shape: same six MxNxK shapes as exp-grouped-gemm-20261006
baseline: aligned GROUP_M=1 versus runtime group4/group8 on the same compiled object
dtype: same three finite dyadic FP16 inputs and FP32 CPU oracles from the original group experiment
measurement: 10 balanced-order rounds, bracketed A/A, 20 repeated dispatches, synchronized wall and HIP events; 64 MiB reset
status: completed
---

## Question and canonical binding

exp-gemm-alignment-stages-20261006 showed that truthful pointer alignment changes vectorization,
resource use and loop lowering. This follow-up tests whether the earlier grouping choices survive
that change. It does not overwrite the unaligned experiment or reclassify its results.

The existing grouped harness now accepts alignment and reference inputs only during CPU preparation.
prepare.jsonl owns the binding; both device entrypoints read it. Input shape/dtype checks and mapping
bijection checks pass. It reuses the original inputs and CPU oracle read-only, instead of maintaining
another copy of the runner or input generator. Every call enforces A/B/C data_ptr()%16==0.

M/N/K, dtype, BM/BN/BK=64/64/32, four execution groups, stages2 and matrix_instr_nonkdim16 stay fixed.
G is runtime i32. Within each shape, G1/G4/G8 share one compiled object and identical recorded resources.
The first five shapes use aligned vectorized code; the odd-stride tail retains the scalar path.
TTIR carries the three pointer attributes; profile reports 8KiB LDS/60 allocated VGPR for regular
shapes, 16KiB/72 for the tail, and zero scratch. Effective waves_per_eu remains1.

Device: bw1100-1/node4 HCU4, gfx938/wave64. Image locator3ad0ae7192b8, Torch2.11.0,
vendor Triton3.6.0, gateway bw1100-bench@77a2848. The source commit is e090b412.

## Acceptance

Two timing runs each pass54 exact oracle checks,276,637,248 output comparisons and240 samples.
Each includes18 successful pointer-alignment checks. The two profile passes each accept108 target
kernel observations, with three distributions and both orderings. Grid, workgroup, wave size,
shape ordering and unchanged within-shape resources are checked by the retained verifier.
All four device phases completed and released (VRAM0, no owned container or visible KFD).

The initial SSH upload failed; a subsequent read confirmed the new directory absent before staging.
No experiment was restarted on inference from a lost connection.

The same finite dyadic-domain limitation remains. Timing is 20-dispatch amortized wall, excludes
transfers/reset/oracle and does not measure a framework call. Single-dispatch profiles have their
own per-dispatch64MiB reset. Full cache eviction and physical exclusivity are not proven.

## Confirmation and historical comparison

Ratios are medians of per-round bracketed G1 / candidate wall. Columns showing the old ratio refer
to a different frozen lowering contract, not a simultaneous factorial experiment.

| M×N×K | aligned G1 wall μs | aligned G8 wall μs | aligned first G8 ratio | aligned confirm G8 ratio | prior unaligned confirm ratio |
|---|---:|---:|---:|---:|---:|
| 512×512×512 | 12.9609 | 12.9240 | 1.0043 | 1.0032 | 1.0092 |
| 2048×2048×512 | 62.4223 | 62.3156 | 1.0018 | 1.0022 | 0.9804 |
| 4096×4096×1024 | 446.0462 | 411.6601 | 1.0895 | 1.0840 | 1.0355 |
| 4096×1024×1024 | 112.2569 | 111.9287 | 1.0033 | 1.0037 | 0.9916 |
| 1024×4096×1024 | 119.1882 | 113.2615 | 1.0510 | 1.0523 | 1.0317 |
| 1088×1025×513 | 60.9316 | 60.6946 | 1.0042 | 1.0040 | 1.0044 |

Large-square confirm forward/reverse ratios are1.0844/1.0835; G4 is1.0613 in both orders.
Wide-rectangle G8 forward/reverse are1.0523/1.0513; G4 confirm is1.0363.
The prior2048-square slowdown was not reproduced in the aligned path; the new0.22% result is too
small to promote as a win. Likewise the narrow rectangle's0.37% is marginal, not a universal reversal.
512 A/A has0.804 outlier; its tiny differences are not stable guidance. All A/A/order samples remain.

## Counter evidence

Means of six reset-balanced target observations per shape/G; FETCH_SIZE is the collector's KiB.

| M×N×K | G1 fetch | G4 fetch | G8 fetch | L2CacheHit G1→G8 |
|---|---:|---:|---:|---|
| 512×512×512 | 1029.10 | 1029.10 | 1029.10 | 0.7568→0.7568 |
| 2048×2048×512 | 4147.35 | 4234.32 | 5004.68 | 0.8245→0.8109 |
| 4096×4096×1024 | 251442.53 | 131773.56 | 73641.03 | 0.7670→0.8731 |
| 4096×1024×1024 | 10348.15 | 10355.47 | 10999.33 | 0.8768→0.8728 |
| 1024×4096×1024 | 48037.86 | 29856.41 | 18341.51 | 0.7969→0.8668 |
| 1088×1025×513 | 2166.86 | 2183.09 | 2214.12 | 0.9027→0.8730 |

L2CacheHit uses the same image/collector fraction scale established by simultaneous raw counters
in exp-grouped-gemm-20261006; that settled calibration was not repeated. No write/total-bandwidth
claim is made from this round's fetch-only pass. Cross-lowering hit-rate comparisons need care:
vectorization changes request generation, so a percentage alone does not explain the time delta.

The large-square/width case gives directionally consistent reuse evidence. The2048 case still reads
more underG8 but no longer shows the old measurable slowdown. That is a counterexample to a direct
rule mapping increased fetch bytes to proportional time loss.

No promotion. This supports revalidating scheduling choices after a lowering/resource change,
not a universalG8 rule, a calibrated cost model or a Cake performance claim.
