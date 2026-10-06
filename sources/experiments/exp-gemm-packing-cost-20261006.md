---
id: exp-gemm-packing-cost-20261006
title: Full-call packing cost and storage-only workspace reuse
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, copy, host-overhead, paired-timing, correctness, negative-result]
confidence: experimental
date: '2026-10-06'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-gemm-packing-cost-20261006
artifacts:
- gemm_packing_cost_probe.py
- gemm_view_precision_probe.py
- prepare.jsonl
- compiled
- measure.jsonl
- confirm.jsonl
- profile-counters.csv
- profile-counters-validation.json
- profile-counters-checks.jsonl
- measure-admission-terminal.json
- confirm-admission-terminal.json
- profile-counters-admission-terminal.json
- profile-admission-terminal.json
- profile.log
- trace-options.txt
- analyze.py
- verify_evidence.py
- accepted-analysis.json
source_commit: bcb07b42
compiler: native vendor Triton3.6.0 with generic/aligned variants; no Cake change
dtype: exact dyadic FP16 inputs and FP32 output; same CPU oracle as original grouped GEMM
shape: 512x512x512,2048x2048x512,4096x4096x1024,1088x1025x513; base/A-offset/C-offset/all-offset
baseline: generic-pointer full call directly on each supported contiguous view
measurement: 10 balanced rounds, generic A/A brackets,20 full calls per sample, HIP events and synchronized wall,64MiB reset
status: completed
---

## What is compared

Source bcb07b42 uses the shared packed_call helper extracted from the view probe; the old
view experiment stays frozen at7f8abf2a. Three strategies use the same GEMM math/group8:
- generic: contiguous-view admission and the generic-pointer kernel;
- pack-dynamic: clone only inputs that need packing, allocate a temporary output when needed,
  invoke aligned kernel, then copy back to the original output view;
- pack-workspace: same per-call data movement, but caller-owned temporary storage is allocated
  once outside timing. Workspace contents are refreshed every call; it is not an input cache.

Three distinct input patterns update the same view objects and workspace before each method is
checked. This catches a stale packed-input cache. Each method must match the original exact CPU
oracle and preserve input storage/output guards. C and inputs are disjoint; one current stream,
fixed shape/view identity and retained temporary references are part of this bounded protocol.
It is not a concurrent, arbitrary-alias or full-framework adapter.

Device: bw1100-1/node4 HCU4, gfx938/wave64; image locator3ad0ae7192b8, Torch2.11.0,
vendor Triton3.6.0; gateway77a2848. BM/BN/BK=64/64/32,4 execution groups,stages2.
A/B/C base views have the source's16-element guard prefix; “base” means legal guarded view,
not the raw starting address of a fresh allocation. Offset cases add one element.
The baseline deliberately omits all pointer attributes; per-operand specialization is not tested.

## Acceptance and interval

Each normal run passes144 changing-input checks and640 samples; a second fixed-source run confirms.
The usable kernel-counter pass accepts48 target GEMM rows, bound to case order/grid/workgroup/waves.
Its full CSV has312 rows including120 copyBuffer kernels; setup/validation are included in that
whole-worker count, so it is not a per-method copy count or copy-time decomposition.
Three successful device phases completed/released. A separate failed HIP-trace phase also released.

Each sample resets64MiB then synchronizes, invokes20 complete calls, records an event span and
synchronized host wall, and releases the last temporary references within the wall interval.
Prior-sample references are released before the next timer starts. The JSON field “dispatches”
counts20 GEMM calls; extra copy/fill work is included in time but is not included in that count.
Guards, per-call copies, temporary allocation/release, compute and output copy-back are included.
Input generation/H2D/oracle/reset and one-time workspace allocation are excluded.
Workspace validity and amortization across arbitrary callers are not established. Complete cache
eviction and physical exclusivity are not proven. Profiled durations are not speed scores.

## Confirmed complete-call results

Representative all-offset cases,μs. Ratios are per-round generic/candidate medians.

| M×N×K | generic | pack-dynamic | pack-workspace | dynamic ratio | workspace ratio | workspace bytes |
|---|---:|---:|---:|---:|---:|---:|
| 512×512×512 | 27.52 | 61.24 | 38.76 | 0.449 | 0.710 | 2097152 |
| 2048×2048×512 | 136.95 | 107.38 | 105.40 | 1.276 | 1.301 | 20971520 |
| 4096×4096×1024 | 958.01 | 551.94 | 551.21 | 1.736 | 1.738 | 83886080 |
| 1088×1025×513 | 60.93 | 83.09 | 81.94 | 0.734 | 0.744 | 6628738 |

First-run ratios for those rows are dynamic0.441/1.273/1.734/0.733 and
workspace0.703/1.296/1.738/0.746. Both orders preserve the large effects.
Small all-offset packing loses even with workspace reuse. Large regular GEMM retains a net benefit.
The odd-stride tail's aligned kernel did not gain vectorization in the prior experiment, and
packing here adds work without a corresponding compute win.

For512 A-offset, dynamic31.26 and workspace25.63 versus generic27.31; C-offset is39.11/25.58
versus27.47. These smaller gains need the A/A limits (roughly0.95–1.01) retained; no universal
threshold follows. The512 base case has an A/A outlier near0.719; do not hide it.
All16 layout/shape results, medians, order split and raw samples are in accepted-analysis.json.

Repacking also changes buffer placement. Notably, large “base” aligned calls are slower than some
repacked layouts despite making no copies. This experiment does not isolate the contribution of
address placement/cache behavior, so do not subtract an unrelated kernel-only timing to infer exact
copy overhead. It reports the combined full-call strategies, not the best possible no-pack kernel.
A per-operand alignment route may retain facts about A/B when only C is misaligned and needs testing.

## Collector limitation

The attempted --hip-trace path failed: the configured libroctracer_tool.so could not be preloaded,
and postprocessing tried missing /opt/dtk-26.04/bin/rocminfo. Its terminal is not_qualified, exit1;
VRAM0/no own container/KFD were observed. No image repair or synthetic trace was performed.
The later counter-only pass is a different, already supported capability and has its own accepted
CSV/receipt. HIP API/memcpy timing attribution remains unavailable; do not promote kernel counters
into a claim that the full API trace worked.

No promotion. The result supports complete-call accounting and storage-only workspace reuse under
its fixed caller contract. It does not select a universal packing rule, modify Compiler policy,
qualify arbitrary numerical inputs, or claim end-to-end model performance.
