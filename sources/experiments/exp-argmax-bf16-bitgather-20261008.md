---
id: exp-argmax-bf16-bitgather-20261008
title: BF16 key-width timing against a bit-preserving Torch argmax and INT16 gather composition
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, bf16, correctness, paired-timing, profiling, reduction]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: archive:wiki-argmax-bf16-bitgather-20261008
artifacts:
- argmax_bf16_gather_probe.py
- binding.json
- composite-cpu-audit.json
- machine-audit.json
- compiled
- prepare.log
- pmc.txt
- profile.log
- profile.jsonl
- profile.csv
- profile-validation.json
- profile-admission-terminal.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- profile-analysis.log
- run.jsonl
- run.log
- run-admission-terminal.json
- confirm.jsonl
- confirm.log
- confirm-admission-terminal.json
- analyze.py
- analysis.json
- analysis.log
source_commit: 97733001
compiler: vendor Triton3.6.0 Gluon; frozen BF16 key32/key64 bodies from b84dedbd
dtype: original selected BF16 bits and int64 first indices; fixed out buffers
shape: M63/4097 crossed with N129/1024; finite/zeros/subnormal/special; same16 frozen inputs
baseline: actual Torch BF16 argmax plus INT16 same-storage bit-view gather, both calls included; direct native key64/key32 pair
measurement: two six-round ABA/BAB batches in eager and graph modes, eight complete calls per block; separate instruction profile and per-cell A/A
limitations:
- Fixed preallocated outputs and resident contiguous inputs; not default allocation, arbitrary stride, storage rebinding or model serving
- Native key encoding covers a wider index range than the two tested row lengths; no device qualification for other lengths
- Warm cache protocol includes synchronized64MiB reset but full cache eviction and physical exclusivity remain unproved
- Framework comparator is a two-primitive composition, not torch.max and not a claim of the fastest available library
status: completed
---

The logical archive ID belongs to the experiment owner. Public records omit internal hosts and account paths.
The CPU-only predecessor is archive:wiki-argmax-bf16-gather-20261008, source55ab0226, with prepare.log and diagnostic source619498a0/bf16_gather_cpu_diagnose.py, cpu-diagnosis.json and cpu-diagnosis.log.
The earlier GPU torch.max failure remains in exp-argmax-bf16-key-20261008.

## Preserve the output contract while constructing a qualified framework comparator

The contract remains first NaN if present, otherwise first numeric maximum, with zero signs tied.
The output must preserve the selected original BF16 bits and return the first index as int64.
No input, oracle, tolerance or special-value case is removed.
The two native kernels import the frozen previous bodies; all eight assemblies match their predecessors.

[torch.argmax](https://docs.pytorch.org/docs/2.11/generated/torch.argmax.html) provides the index and
[torch.gather](https://docs.pytorch.org/docs/2.11/generated/torch.gather.html) retrieves the selected elements.
The initial composition used BF16 gather. CPU qualification failed before any GPU work:
all indices were correct, but5,544 NaN rows changed to0xffff across the four special cases.
CPU diagnostic totals are41,41,2731,2731 value-bit mismatches for63×129,63×1024,4097×129,4097×1024.
The same indices with INT16 bit-view gather preserve all original bits.

The accepted successor therefore runs:

```python
torch.argmax(bf16_input, dim=1, out=int64_indices)
torch.gather(int16_input_view, 1, index_view, out=int16_output_view)
```

The INT16 views alias the existing BF16 storage. They do not numerically convert BF16 values to integers.
Ordering is still computed from BF16 input; only the selected two-byte payload is copied through the integer view.
The public value output is the BF16 view of that output storage.
Views are constructed in setup under the fixed-out contract; both device operations and their submissions are inside each measured call.
The failed BF16 gather predecessor has no GPU admission or performance result.

The upstream v2.11.0 [ScatterGatherKernel.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/ScatterGatherKernel.cu)
uses an opaque element representation for assignment paths. The installed package omits that implementation header, so upstream code is a mechanism reference rather than proof of the installed build.
Actual device trace below supplies the local route evidence.

## CPU and complete device qualification

All16 CPU bit-view composition cases match the original BF16 bits and indices.
Profile bw-e44a2938f149 passes1,408 target dispatches in the exact declared stage order:
352 ArgMaxOps,352 gather,352 native key64 and352 native key32 calls.
It also passes96 refreshed-input checks and12 first graph replays; all12 graphs have explicit release records.
Special values are included in both eager and graph checks, closing the earlier incomplete graph boundary.

The observed framework reduction is ArgMaxOps<float>, and its gather instantiates OpaqueType<2>.
Its comparator handles index selection separately from the raw two-byte value retrieval.
This route is distinct from the earlier MaxOps value/index reduction; the failed torch.max result remains unchanged.

Run bw-bb328f9d63d0 and confirm bw-d6fbb57175ea each pass96 refreshed-input checks,12 first replays and1,728 timing blocks.
Each fixed-output block performs eight complete calls; outputs are poisoned before tested execution and checked against the unchanged oracle afterward.
Input and output guards remain unchanged. This is fixed-output qualification, not fresh-result lifetime qualification.

Runtime: HCU3/gfx938/wave64, gateway77a2848, DTK image locator3ad0ae7192b8, Torch2.11.0, vendor Triton3.6.0.
All three GPU jobs complete with exit0, after_vram0%, no visible KFD user and no remaining task container.
Per-user serialization still does not establish physical exclusivity.

## Actual resource and instruction exchange

For M4097, finite input, graph profile:

| N | key | actual VGPR | SGPR | LDS / scratch | SQ_INSTS_VALU per complete call |
|---|---|---:|---:|---|---:|
| 129 | uint64 | 16 | 16 | 0 / 0 | 434558 |
| 129 | uint32 | 16 | 16 | 0 / 0 | 332061 |
| 1024 | uint64 | 32 | 16 | 0 / 0 | 1119258 |
| 1024 | uint32 | 56 | 16 | 0 / 0 | 934761 |

Each native route launches4,100 waves in these large cases.
The previous static observation is now complemented by actual allocation: the long-row compact key uses more physical VGPR, while its VALU total is lower.
Static DPP sites remain12→6. Neither route spills or allocates LDS in the observed profile.
Do not infer an occupancy limit from register counts alone; this record does not identify a single limiting resource.

Framework raw VALU totals include both stages:664,023 for N129 and2,022,173 for N1024.
Gather contributes1,449 instructions and20 waves in each large case.
The analysis stores normalized VALUInsts/LDSInsts separately by stage. It sums raw counts across kernels, not per-wave averages.

## Direct paired timing

Three pairs are measured independently: key64/key32, framework/key64, framework/key32.
Six ABA/BAB rounds cover all16 inputs in both eager and graph modes, with per-cell A/A and reversed confirmation ordering.
HIP events are initialized before timing, and every timed block follows the same synchronized64MiB reset.
Profile durations are not used for speed scores.

Finite confirmation results in μs per complete call; each row is its own pair:

| M,N | mode | pair | baseline | candidate | paired ratio |
|---|---|---|---:|---:|---:|
| 63,129 | eager | key64/key32 | 15.89150 | 15.91025 | 0.9979 |
| 63,1024 | graph | key64/key32 | 8.43200 | 8.30825 | 1.0162 |
| 4097,129 | eager | key64/key32 | 15.23788 | 15.30400 | 0.9970 |
| 4097,129 | graph | key64/key32 | 12.79175 | 11.94050 | 1.0686 |
| 4097,1024 | eager | key64/key32 | 22.35238 | 20.34625 | 1.1011 |
| 4097,1024 | graph | key64/key32 | 21.46363 | 19.56125 | 1.0981 |
| 4097,129 | graph | framework/key32 | 23.23850 | 11.81675 | 1.9598 |
| 4097,1024 | graph | framework/key32 | 44.74225 | 19.58875 | 2.2842 |

Across all four patterns, largeN129 key64/key32 graph ratios span1.085–1.089 in run and1.069–1.086 in confirm.
Its eager ratios stay near1, including regressions; small-batch differences also remain small.
LargeN1024 graph spans1.089–1.097 and1.096–1.100; eager spans1.083–1.092 and1.095–1.101.
The wider VGPR allocation therefore does not prevent a bounded measured improvement.

All wall A/A ranges remain: run0.853–1.114, confirm0.917–1.105.
For finite large key64/key32 graph pairs, confirm A/A maxima are1.0108 at N129 and1.0159 at N1024.
No sample is trimmed, no pair is replaced, and no cross-round absolute-time ratio is used.
The approximately2.28× framework ratio includes the whole two-stage composition; it is not an acceleration claim against single torch.max.

## Disposition

No promotion. The same-contract BF16 key-width comparison now has complete numerical, graph, profile and paired evidence.
A smaller key reduces selected communication/ALU work but does not guarantee lower registers or benefit at every caller boundary.
Raw payload retrieval through same-storage integer views is a qualified local framework composition here, not a blanket claim about every dtype/backend.
The BF16 gather CPU failure and earlier torch.max GPU failure remain linked. No Compiler, Target or default library route was changed.
