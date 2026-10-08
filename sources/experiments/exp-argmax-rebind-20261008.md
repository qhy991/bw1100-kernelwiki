---
id: exp-argmax-rebind-20261008
title: Per-call current-storage anchor binding preserves argmax correctness but consumes most of the caller gain
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, fp32, correctness, paired-timing, profiling, runtime-guard]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: archive:wiki-argmax-rebind-20261008
artifacts:
- argmax_rebind_probe.py
- binding.json
- machine-audit.json
- compiled
- prepare.log
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- run.jsonl
- run.log
- run-admission-terminal.json
- confirm.jsonl
- confirm.log
- confirm-admission-terminal.json
- analyze.py
- analysis.json
- analysis.log
source_commit: 876c3e1e
compiler: vendor Triton3.6.0 Gluon; frozen peeled body from cc3cf41c; CPU binder from ec0d1a50
dtype: original FP32 selected bits and int64 first indices; fresh retained outputs
shape: M63/4097 crossed with N129/1024 and pointer offsets0/1; finite/zeros/subnormal/special
baseline: actual Torch default max and same peeled kernel with anchor binding outside timing
measurement: three independent comparisons; two six-round ABA/BAB batches, eight retained outputs per timed block; separate marked instruction profile
limitations:
- Fixed shape and contiguous FP32 only; offsets2/3, arbitrary strides, autograd and multiple streams remain unqualified
- Input allocation and storage replacement are outside timing; current-view validation and anchor reconstruction are inside every rebound call
- Warm allocator and resident inputs; old-result destruction outside timing; physical exclusivity and complete cache eviction unproved
- Collected instruction metrics do not establish cache or occupancy causality
status: completed
---

The logical archive ID resolves through the experiment owner. Internal host and account paths are omitted from this public record.

## Close the device and caller boundary left by the CPU preflight

exp-argmax-anchor-cpu-20261008 showed that an old anchor retains old storage after Tensor.set_.
exp-argmax-peel-20261008 measured the peeled kernel with an anchor bound during fixed-input setup.
This successor preserves both records and asks what happens when current-input binding enters every call.

The three arms are actual Torch max with default output allocation, prebound peeled argmax, and rebound peeled argmax.
The native arms import the same frozen kernel body. Eight compiled assemblies match the preceding source exactly.
Both allocate new FP32 values and int64 indices from the same metadata templates.
No output storage is reused while its results remain live.

The rebound arm calls the CPU-qualified binder on each invocation. It checks dtype, fixed shape, contiguity and pointer remainder.
It builds `x.view(-1)[shift:]`, checks the resulting alignment/address relation, and launches the specialized kernel.
The measured added cost includes those checks and view construction. It is not a measurement of one view operation alone.
The prebound arm receives the same current anchor, but constructs it outside timing.

## Changed storage and retained-output checks

Each input refresh allocates a guarded parent while the old parent is alive, then changes the same Tensor object's storage.
Its identity remains fixed and its data pointer changes. The same per-shape Tensor object is also reused across offset0/1 cases.
The inputs and exact output oracles are inherited from exp-argmax-fp-key-20261008.
All values retain their original FP32 bits, including zero sign and NaN payload; all indices use int64.

Profile job bw-5c5b0ee46d8d passed 40 storage refreshes, 96 correctness blocks and 768 output pairs.
Run bw-46ae19633f20 and confirm bw-51bd1f2c43f9 each passed 72 refreshes, 96 correctness blocks and 1,728 timing blocks.
Across both timing batches, 1,536 fresh-check output pairs and 27,648 timed output pairs were checked.
Each check covers all eight returns, current input/guards, nonoverlapping storage and the prior retained result blocks.

Input allocation, copying test data and set_ belong to test setup and remain outside timing in all arms.
The rebound arm still reconstructs the anchor on all eight invocations inside every timed block.
This is not a benchmark of arbitrary external input ingestion or allocation.

## Actual profile shows no extra device work in the observed call scopes

The independent profile contains 1,152 target dispatches in 144 marked blocks.
Each block contains exactly eight expected reductions and no extra copy or helper kernel.
For every one of the 32 case-pattern controls, prebound/rebound have identical measured Wavefronts, SQ_INSTS_VALU, VALUInsts and LDSInsts.
The frozen assemblies and observed launch geometry also agree.
These observations support a host-side wrapper cost; they do not measure every possible counter.

The target remains HCU3/gfx938/wave64, gateway77a2848, DTK image locator3ad0ae7192b8, Torch2.11.0 and vendor Triton3.6.0.
All three admissions completed with exit0, after_vram0%, no visible KFD user and no surviving task container.
Per-user serialization does not establish physical exclusivity.

## Complete caller cost reduces the apparent gain

The comparisons are prebound/rebound, Torch/prebound and Torch/rebound, each measured directly.
Six ABA/BAB rounds and A/A controls are retained for every shape, offset and input pattern.
The confirmation batch reverses pattern order and alternates comparison order.
HIP events are initialized before timing; each timed block follows a synchronized 64MiB reset.
Marker calls occur only in profile.

For finite inputs, the confirmation batch gives these per-call wall medians. Rows use their own independent pairs.

| M,N,offset | pair | baseline μs | candidate μs | paired baseline/candidate |
|---|---|---:|---:|---:|
| 63,129,0 | Torch / rebound | 19.63500 | 37.58013 | 0.5255 |
| 63,1024,1 | Torch / rebound | 19.46625 | 36.98638 | 0.5248 |
| 4097,129,1 | Torch / rebound | 23.02725 | 37.10763 | 0.6207 |
| 4097,1024,0 | prebound / rebound | 25.53088 | 36.98763 | 0.6920 |
| 4097,1024,0 | Torch / prebound | 39.95000 | 25.49338 | 1.5662 |
| 4097,1024,0 | Torch / rebound | 39.94500 | 36.94525 | 1.0836 |
| 4097,1024,1 | prebound / rebound | 28.93688 | 37.36888 | 0.7730 |
| 4097,1024,1 | Torch / prebound | 43.41475 | 28.90438 | 1.5008 |
| 4097,1024,1 | Torch / rebound | 43.45475 | 37.32263 | 1.1634 |

Across all input patterns, the direct prebound/rebound submit difference is 13.109–14.039μs in run and 12.617–13.339μs in confirm.
This cannot be subtracted mechanically from wall time because submission and device execution overlap.
For M4097/N1024 offset1, Torch/rebound spans 1.139–1.167 in run and 1.154–1.167 in confirm.
Offset0 spans only 1.056–1.067 and 1.064–1.084 respectively. Small M and N129 remain slower than Torch.
The prebound gain remains visible within this same experiment, so the reduction is not inferred by dividing results from different rounds.

All wall A/A controls are retained: run range0.865–1.129 and confirm0.870–1.118.
For finite large1024 Torch/rebound, confirm A/A maxima are1.0767 at offset0 and1.0721 at offset1.
The small aligned-input advantage must be read with this noise; it is not a broad production speed claim.
No sample was trimmed and no run was extended after seeing results.

## Disposition

No promotion. Rebinding preserves the tested device results but substantially consumes the fixed-binding caller gain.
Do not cache a data anchor solely by Tensor object identity, and do not inherit the approximately1.50× fixed-binding ratio for a per-call adapter.
A lighter host path or a storage-aware cache is a future hypothesis requiring its own lifetime, invalidation and complete-call checks.
No Compiler, Target fact or default Torch replacement was changed.
