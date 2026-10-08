---
id: exp-argmax-bf16-inline-20261008
title: A pure gfx938 inline max loses recognized reduction fusions without reducing register allocation
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, assembly, bf16, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: archive:wiki-argmax-bf16-inline-qualified-20261008
artifacts:
- argmax_bf16_inline_probe.py
- binding.json
- compiled
- prepare.log
- backend-stages.txt
- audit_inline.py
- inline-audit.json
- pmc.txt
- profile.jsonl
- profile.csv
- profile.log
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
source_commit: 1cac1ca7
compiler: vendor Triton3.6.0 Gluon with native uint32 max or pure gfx938 v_max_u32 inline template
dtype: selected original BF16 bits and int64 first indices; fixed output buffers
shape: M63/4097 crossed with N129/1024; same16 finite/zeros/subnormal/special inputs
baseline: frozen key32 body and derived builtin bridge; three direct pairs separate bridge from inline effect
measurement: two six-round ABA/BAB batches in eager and graph modes, eight complete calls per block; separate instruction profile and per-cell A/A
limitations:
- Two batches exhibit very different absolute-time regimes; device-state cause is unknown and times are not pooled
- Fixed resident inputs and outputs; no default-allocation, arbitrary-shape or full-model qualification
- Instruction counts and emission changes do not uniquely identify the latency bottleneck or compiler pass
- One pure register-only max template does not qualify memory, synchronization or matrix inline assembly
status: completed
---

The logical archive ID belongs to the experiment owner. Internal hosts and account paths are omitted from this public record.
CPU-only predecessor archive:wiki-argmax-bf16-inline-20261008 retains source6d33ac5e, prepare.log and its partial compiled output.

## Observe the actual code layers and change only the combine boundary

The installed backend is triton.backends.hcu.compiler.HIPBackend.
Its captured add_stages definition gives Triton ttir→ttgir and Gluon ttgir entry routes, followed by llir→amdgcn→hsaco.
The standard definition has no separate PTX stage. The amdgcn artifact declares exact target gfx938; hsaco is the loadable code object.
This is evidence from the installed HCU route, not a claim that upstream AMD LLVM accepts gfx938.
The conceptual layer definitions and public documentation are collected in doc-dtk-code-layers.

The native BF16 key algorithm, S1/N129 and S4/N1024 layouts, masks, outputs and original16 inputs remain unchanged.
Three arms are compared: the frozen previous kernel, a derived builtin bridge, and the same derived body with inline maximum.
The bridge separates function restructuring from the proposed inline change.
The inline reducer is:

```python
gl.inline_asm_elementwise(
    'v_max_u32 $0, $1, $2',
    constraints='=v,v,v', args=[a, b],
    dtype=gl.uint32, is_pure=True, pack=1,
)
```

This template computes a register-only unsigned maximum. The purity claim covers absence of side effects; it does not declare algebraic max semantics to optimization passes.
The v constraints request vector registers; LLVM still assigns their physical names.
The code has no hidden memory operation, barrier or cross-lane operation inside the template.
Cross-lane reduction remains owned by Gluon lowering.

## Preserve the CPU interface failure separately

The first source passes a reducer function as a constexpr parameter to gl.reduce.
The vendor implementation tries to call that constexpr wrapper as a JIT function and raises an AttributeError because it has no fn attribute.
The failure occurs in the builtin bridge before the inline candidate is compiled.
It therefore does not establish that v_max_u32 or inline assembly is unsupported.
Only the first frozen control is emitted; the predecessor has no GPU admission.

Source1cac1ca7 uses a compile-time boolean to choose explicit builtin and inline reducer names.
It compiles12 instances in a new directory. Four frozen controls retain their exact preceding assembly.
The derived builtin matches the frozen instruction/branch/resource-directive view for all four shapes after normalizing the kernel symbol name.
That view comparison excludes debug/source-location metadata and is not an HSACO byte-identity claim.

## Recognized IR and opaque inline instructions produce different code

Builtin LLVM IR contains llvm.umax.i32 calls. Inline LLVM IR contains calls to the native assembly template instead.
In the emitted assembly, six DPP sites remain in either route, but their operations differ:

| N | builtin LLVM umax sites | inline asm sites | builtin target max instructions | inline target max instructions |
|---|---:|---:|---|---|
| 129 | 8 | 9 | 3 ordinary max + 5 max-with-DPP | 9 ordinary max |
| 1024 | 21 | 21 | 2 ordinary max + 7 three-input max + 5 max-with-DPP | 21 ordinary max |

The builtin forms include v_max3_u32 and v_max_u32_dpp. The inline form does not combine its template with those operations in this build.
DPP exchange instructions remain separate from the inline max operations.
Differences elsewhere in generated dataflow also remain; the full dynamic increase is not attributed solely to counting the missing fused sites.

## Device semantics and dynamic work

Profile bw-7b015d914ca4 verifies1,056 target dispatches,96 refreshed-input checks and12 first graph replays, with12 explicit graph releases.
All original NaN bits, zero signs and first indices pass in eager and graph modes, with unchanged input and guards.
The frozen and builtin bridge have equal raw instruction controls across all32 shape/pattern/mode combinations.

For M4097, finite input and graph profile:

| N | method | SQ_INSTS_VALU per call | actual VGPR | SGPR | LDS / scratch |
|---|---|---:|---:|---:|---|
| 129 | frozen / builtin | 332061 | 16 | 16 | 0 / 0 |
| 129 | inline | 360761 | 16 | 16 | 0 / 0 |
| 1024 | frozen / builtin | 934761 | 56 | 16 | 0 / 0 |
| 1024 | inline | 1045461 | 56 | 16 | 0 / 0 |

Every large-case route launches4,100 waves. Inline increases observed VALU work without reducing resource allocation.
No unique occupancy, cache or dependency-stall cause is inferred from these counters.

## Two timing regimes, retained separately

Run bw-4563c0a3fb35 and confirm bw-a86600dada4c each pass96 refreshed-input checks,12 first replays and1,728 timing blocks.
The fixed protocol retains six ABA/BAB rounds for frozen/builtin, builtin/inline and frozen/inline in both eager and graph modes.
The confirmation reverses pattern and comparison order; all inputs, samples and A/A observations remain.
HIP events are initialized before timing and each block follows a synchronized64MiB reset.

The first confirmation SSH attempt resets during handshake. A read-only check finds no confirmation log or admission before the not-yet-started batch is submitted.
No accepted job is restarted because of an observer failure.

Finite large-case graph observations, per-call medians in μs:

| batch | N | builtin wall | inline wall | paired builtin/inline | builtin HIP-event | builtin submit |
|---|---|---:|---:|---:|---:|---:|
| run | 129 | 38.81500 | 39.91250 | 0.9739 | 31.37800 | 2.03988 |
| confirm | 129 | 11.85425 | 12.13800 | 0.9782 | 7.89950 | 1.79613 |
| run | 1024 | 71.00925 | 77.05638 | 0.9224 | 63.41575 | 2.08238 |
| confirm | 1024 | 19.59625 | 20.78113 | 0.9448 | 15.67888 | 1.88863 |

The absolute-time change also occurs inside the device-event boundary, not only host submission.
That event interval covers the complete graph execution boundary; it is not an isolated instruction-latency measurement.
Both receipts name HCU3 and the same image. Source, binding and command options remain fixed.
No historical clock or exclusive-occupancy trace explains the change, so its device-state cause is unknown.
The two absolute regimes are not pooled or compared as an optimization result, and neither is discarded.

Within each batch, large frozen/builtin controls remain near1, about0.995–1.003 across the reported cells.
For all four patterns at largeN1024, builtin/inline graph ratios are0.922–0.923 in run and0.939–0.945 in confirm: inline is slower in both regimes, with different magnitudes.
At largeN129, graph ratios are0.969–0.974 and0.971–0.978. Eager and small-batch differences are less consistent; some cells reverse.
This does not establish a universal inline penalty or a stable cross-run absolute latency.

All wall A/A ranges remain: run0.855–1.134; confirm0.782–1.350.
Finite largeN1024 builtin/inline graph confirm has A/A maximum1.0078, beside its0.9448 paired ratio.
No third batch is added to select a preferred regime, and no samples are trimmed.

All three GPU admissions complete with exit0, after_vram0%, no visible KFD user and no surviving task container.
Target gfx938/wave64, gateway77a2848, DTK image locator3ad0ae7192b8, Torch2.11.0, vendor Triton3.6.0.
Per-user serialization does not prove physical exclusivity or stable clocks.

## Disposition

No promotion. Keep the builtin path for this measured domain and retain the inline experiment as a counterexample.
A lower-level spelling can remove recognized operation semantics and miss useful code generation; inspect LLIR, emitted ISA and dynamic work instead of assuming direct assembly is faster.
The observed large-case regression is bounded by the separate timing regimes and the declared fixed-output contract.
No Compiler, Target, general inline API policy or default framework route was changed.
