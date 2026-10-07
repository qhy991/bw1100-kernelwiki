---
id: exp-graph-caller-20261007
title: Single-GEMM graph replay loses its advantage when caller copies are required
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, host-overhead, paired-timing, profiling, gemm, copy, negative-result]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-graph-caller-qualified-20261007
artifacts:
- graph_caller_probe.py
- graph_replay_probe.py
- matrix_instruction_probe.py
- binding.json
- prepare.log
- profile.jsonl
- profile.csv
- profile-validation.json
- profile-admission-terminal.json
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
source_commit: bf9b14bf
compiler: frozen native vendor Triton3.6.0 GEMM, MMAC16,4waves,stages2
shape: 512x512x512 and4096x4096x1024
baseline: direct caller-buffer GEMM versus workspace eager and workspace graph with mandatory input copy and output copy-back
dtype: FP16 inputs, FP32 accumulator/output; three exact dyadic CPU-oracle patterns
measurement: 12 rotating single-GEMM calls per timed sample,12 bracket rounds; allocation/build/reset/checks outside timing
limitations:
- Fixed shapes and three long-lived caller storage sets, not arbitrary allocation churn
- Caller and workspace phase0 but separate physical allocations
- No production graph cache, dynamic shape, concurrency or end-to-end serving qualification
- Excludes one-time allocation/capture costs; reports them separately where observed
status: completed
---

## Why a new caller boundary

exp-graph-replay-20261007 measured20 repeated kernels with fixed addresses already resident in the graph.
That result does not establish a win when each logical call uses a different caller buffer.
This successor keeps one GEMM per call and rotates among three disjoint caller A/B/C sets.
All three input distributions have different CPU oracle outputs; the source checks distinct pointers and oracles.

Three strategies share the same original kernel, tile64×64×32,G8,MMAC16,4wave64,stages2:

- direct launches on current caller A/B/C.
- workspace-eager copies current A/B into fixed workspace, launches eager GEMM, then copies output to caller C.
- workspace-graph performs the same copies and copy-back, replacing only the workspace GEMM launch with graph replay.

Workspace lifetime and allocation are outside timing; data is refreshed on **every call**, not cached by contents.
Copies are not captured: the graph holds one GEMM over workspace addresses. Source pointer rotation occurs outside the graph.
This is a bounded fixed-shape strategy comparison, not a complete framework implementation.

## Single-call opaque structure and qualification

Initial ab686a9e under wiki-graph-caller-20261007 reused the prior20-call assumption of two type200 nodes.
It failed before any accepted timing; its profile receipt remainsnot_qualified/exit1, with release observed.
No original failure log or partial CSV was overwritten.

Successorbf9b14bf records structure first and qualifies actual dispatches dynamically.
Both shapes capture **one type200 node, no edges**. Timing requires the same structure as its accepted profile.
This observation does not decode the vendor node or establish a universal packing capacity.

The profile has two warmup GEMMs, one initial graph replay, and18 strategy/input checks per shape,
so42 target dispatches total are expected. Canonical verifier accepts42ll_grouped_gemm rows among1346total rows.
The first21 have grid16384/workgroup256/Wavefronts256; the next21 have1048576/256/16384.
Task analysis additionally verifies all36caller checks in both strategy orders, three inputs, exact outputs and guards.
Auxiliary copy/fill/compare kernels are not counted as GEMM. Profile is qualification, not performance timing.

## Measurement contract

Environment: bw1100-1/node4 HCU3, gfx938/wave64, Torch2.11.0/vendor Triton3.6.0,
image locator3ad0ae7192b8, gateway77a2848. Per-user admission does not prove physical exclusivity.

All caller and workspace views use phase0 modulo256 and valid16-byte pointers. Separate allocations remain separate;
this coordinate does not imply identical cache placement or physical pages. Inputparents and output guards are retained.
Per sample, three untimed calls warm each input, then caller outputs are poisoned withNaN,64MiB reset is issued,
and synchronization completes before timing. Full cache eviction is unproven.

The timed loop executes12single-GEMM calls and rotates slot=(round+j)%3. Every slot is used four times.
For workspace paths, both input copies and outputcopy-back occur on every timed call.
After completion, all three caller outputs are checked, all inputparent contents are unchanged, and caller/workspace guards hold.
These are three final-output checks per sample, not an independent observation after every one of12dispatches.
Profile provides per-call checks; source/control-flow fixes the timed operations.

There are12rounds per shape, with direct at both ends and the two workspace strategies in alternating order.
The independent process reverses the middle order. Eachrun96samples/288finalcaller-output checks,
tworuns192samples/576finaloutputs; comparator controls and graph.reset/release also verified.
No rounding/tolerance changes were introduced.

## Independent results with copies included

Per-call μs, median of the independent reverse-order confirmation:

| shape | strategy | wall | host submit | device event span |
|---|---|---:|---:|---:|
| 512³ | direct | 18.721 | 9.890 | 15.852 |
| 512³ | workspace-eager | 33.984 | 29.199 | 31.071 |
| 512³ | workspace-graph | 34.255 | 29.642 | 31.505 |
| 4096×4096×1024 | direct | 412.312 | 9.640 | 408.346 |
| 4096×4096×1024 | workspace-eager | 546.319 | 29.141 | 542.244 |
| 4096×4096×1024 | workspace-graph | 547.243 | 26.736 | 543.411 |

First-run wall values are18.716/34.270/33.998μs for512 and412.281/546.094/547.329μs for the large shape.
Workspace-graph costs about1.830×direct on512,1.327×direct on the large shape in confirmation.
Within identical-copy workspace paths, eager/graph paired ratio medians are1.0107 then0.9919 for512:
the tiny direction changes between runs. Large-shape ratios are0.9978 then0.9983, not a graph win.

Direct A/A ranges: small0.9427–1.0159(first),0.9357–1.0127(confirm);
large0.9971–1.0034 and0.9988–1.0039. All samples are retained; no sub-percent improvement is promoted.
Event span includes copies, dispatch gaps and GEMM; it cannot isolate arithmetic kernel duration.

## Payload and setup costs

Required workspace payload is2MiB for512,80MiB for the large shape, excluding guards/allocator overhead.
Each workspace call copies this payload once across A/B input and C output; this is payload volume,
not measured HBM traffic. Reading and writing a copy need not equal one payload's hardware transaction volume.

Confirmation capture/instantiate costs are510.948/112.563μs for512 and417.183/35.638μs for the large shape.
These exclude allocation, CPU compile, other preparation and startup costs. Including them would not create a
net win in this workload, but no universal break-even or production lifetime is inferred.

## Disposition

No promotion. The prior fixed-address20-call graph result and this negative single-call caller result coexist:
graph submission amortization, copies, pointer lifetime and the amount of captured work define different contracts.
Compare against both direct and an identical-copy eager control before recommending a graph strategy.
Do not cache stale input contents or silently return workspace output instead of writing the caller's destination.

This work does not reject HIP graphs in general; larger captured operation sequences, upstream-owned stable buffers,
or capture of other caller operations may have different economics. Those are new experiments, not evidence already established here.
