---
id: exp-argmax-bf16-layout-20261008
title: Contiguous per-thread grouping changes BF16 load width without reducing long-row register allocation
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, bf16, correctness, paired-timing, profiling, tiling, vgpr]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: archive:wiki-argmax-bf16-layout-20261008
artifacts:
- argmax_bf16_layout_probe.py
- binding.json
- machine-binding.json
- compiled
- prepare.log
- audit_machine.py
- machine-audit.json
- pmc.txt
- profile.jsonl
- profile.csv
- profile.log
- profile-validation.json
- profile-admission-terminal.json
- requests-pmc.txt
- requests.jsonl
- requests.csv
- requests.log
- requests-validation.json
- requests-admission-terminal.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- profile-analysis.log
- requests-analysis.json
- requests-analysis.log
- run.jsonl
- run.log
- run-admission-terminal.json
- confirm.jsonl
- confirm.log
- confirm-admission-terminal.json
- analyze.py
- analysis.json
- analysis.log
source_commit: 7758c4eb
compiler: vendor Triton3.6.0 Gluon; frozen BF16 uint32-key body from b84dedbd; size_per_thread column grouping1/2/4/8
dtype: selected original BF16 bits and int64 first indices; fixed out buffers
shape: M63/4097 crossed with N129/1024; same16 finite/zeros/subnormal/special inputs
baseline: frozen previous key32 layout, S1 at N129 and S4 at N1024; three direct candidate pairs per shape
measurement: two six-round ABA/BAB batches in eager and graph modes, eight complete calls per block; independent instruction and TCC-request passes
limitations:
- Fixed resident contiguous input and preallocated outputs; no new framework or default-allocation comparison
- S changes lane/register ownership and generated work together; no unique bottleneck or cache-causality attribution
- Physical exclusivity and complete cache eviction remain unproved; all A/A samples retained
- Tiny long-row differences are not promoted to a default layout
status: completed
---

The logical archive ID is retained by the experiment owner; public records omit internal host and account paths.

## Change grouping, not the key algorithm or the reference layout

exp-argmax-bf16-bitgather-20261008 found a narrower key with more long-row VGPR allocation but lower latency.
This successor keeps that uint32-key body, value/index contract, masks, grid, four wave64 execution groups and options unchanged.
Only the column component of Gluon BlockedLayout.size_per_thread changes to1,2,4 or8.
The declared layout is `[1,S], [1,64], [4,1], order[1,0]`.

The [Gluon BlockedLayout API](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.BlockedLayout.html)
and [layout tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html) describe hierarchical register/lane/warp tiling.
S names a contiguous register subtile; it is not necessarily the total elements held by a lane across the full logical tensor.
For the1024-column row, each tested nonredundant grouping still assigns1024/64=16 logical values per lane.
Changing S can change scheduling and allocation, but does not itself reduce that logical value count.

The baseline remains S1 for N129 and S4 for N1024, exactly as in the qualified predecessor.
All four baseline assemblies match their frozen versions. There are16 compiled layout instances and three direct pairs per shape.
The original16 inputs, exact BF16-bit/int64 oracle and output guards are reused without regeneration.
No cross-round framework ratio is constructed.

## Static emission and actual resource observations

At N1024, all layouts have six DPP sites, one readlane site and no static DS site.
Their payload loads widen with S; every route also reloads the winning ushort:

| S | static load sites, including winner | source VGPR | actual VGPR |
|---|---|---:|---:|
| 1 | 17 ushort | 55 | 56 |
| 2 | 8 dword + 1 ushort | 54 | 56 |
| 4 | 4 dwordx2 + 1 ushort | 54 | 56 |
| 8 | 2 dwordx4 + 1 ushort | 54 | 56 |

All have actual SGPR16, LDS0 and scratch0. The requested grouping change did not reduce physical VGPR allocation in this domain.

At N129, odd row stride and the padded256-column domain do not produce wide payload loads:

| S | static ushort load sites | source / actual VGPR | actual SGPR | DS observation |
|---|---:|---|---:|---|
| 1 | 4 | 15 / 16 | 16 | none |
| 2 | 4 | 16 / 16 | 16 | none |
| 4 | 5 | 18 / 20 | 32 | none |
| 8 | 9 | 30 / 32 | 32 | one ds_swizzle_b32, SWAP16 |

The S8 IR retains the requested layout. Its DS exchange occurs while actual LDS allocation remains0.
A zero LDS allocation is therefore not evidence that the kernel uses no DS communication.
No new bank count, register-pool constant or occupancy limit is inferred.

## Dynamic work and request counts are separate observations

Instruction profile bw-093c3c28858b and request profile bw-5370cde3be6d each contain1,408 target dispatches.
Each pass independently verifies128 refreshed-input checks and16 first graph replays, with16 graph release records.
The two passes agree on actual resource allocation for all16 layout instances.

For M4097, finite input and graph replay, each variant launches4,100 waves. Raw values below are per complete call, averaged over the observed block:

| N | S | SQ_INSTS_VALU | TCC_READ_sum | TCC_WRITE_sum |
|---|---|---:|---:|---:|
| 129 | 1 | 332061 | 18313.875 | 8194 |
| 129 | 2 | 344361 | 18211.750 | 8194 |
| 129 | 4 | 430461 | 18389.625 | 8194 |
| 129 | 8 | 725661 | 18808.375 | 8194 |
| 1024 | 1 | 1008552 | 136384.500 | 8194 |
| 1024 | 2 | 934761 | 103741.000 | 8194 |
| 1024 | 4 | 934761 | 82562.750 | 8194 |
| 1024 | 8 | 934761 | 103284.750 | 8194 |

Instruction totals and TCC requests come from separate passes; they are not simultaneous correlated samples.
S8 has fewer and wider load sites than S4 at N1024, yet more observed TCC read requests.
That observation does not identify HBM bytes, a cache-line size or a unique reason for the difference.
S2/S4/S8 have the same observed VALU total at N1024, but their requests differ.
At N129, S8 raises VALU work substantially and has LDSInsts1 per wave while its LDS allocation stays0.

## Paired timing preserves the negative result

Run bw-e88d7db076de and confirm bw-16f41331fad8 each pass128 refreshed-input checks,16 first graph replays and1,728 timing blocks.
All layouts retain exact selected bits, first indices, input contents and output guards in eager and graph paths.
Six ABA/BAB rounds, per-cell A/A, reverse confirmation order, initialized HIP events and synchronized64MiB resets are unchanged.
The interval includes eight complete fixed-output calls. Profile durations are not speed scores.

Finite-input confirmation, per-call wall μs; each row is an independent pair:

| M,N | mode | pair | baseline | candidate | baseline/candidate |
|---|---|---|---:|---:|---:|
| 63,129 | graph | S1/S8 | 7.76325 | 8.69200 | 0.8961 |
| 4097,129 | eager | S1/S8 | 15.02663 | 22.52850 | 0.6663 |
| 4097,129 | graph | S1/S2 | 11.86925 | 11.97925 | 0.9800 |
| 4097,129 | graph | S1/S4 | 11.89175 | 13.15288 | 0.9025 |
| 4097,129 | graph | S1/S8 | 11.85675 | 22.05613 | 0.5406 |
| 4097,1024 | eager | S4/S1 | 20.36375 | 21.33863 | 0.9537 |
| 4097,1024 | graph | S4/S1 | 19.52125 | 20.32613 | 0.9597 |
| 4097,1024 | graph | S4/S2 | 19.56875 | 19.53250 | 1.0027 |
| 4097,1024 | graph | S4/S8 | 19.52500 | 19.24250 | 1.0122 |

Across all patterns, largeN129 graph S1/S8 spans0.533–0.541 in run and0.539–0.545 in confirm.
S1/S4 remains about0.90; S1/S2 remains about0.98. The wider groupings do not improve this row length.
At largeN1024, S4/S1 stays about0.960–0.973, while S4/S2 is near1.
S4/S8 graph spans1.007–1.015 and1.011–1.015, a small effect rather than a basis for a universal default.
Small-batch eager comparisons are generally near1 and include reversals.

All wall A/A samples remain: run0.877–1.183 and confirm0.857–1.346.
For finite largeN1024 S4/S8 graph, confirm A/A reaches1.0704, larger than its approximately1.012 paired effect.
No sample is removed or extra timing added. Request counts alone do not rank measured latency: S8's request increase coexists with this tiny paired advantage.

All four jobs complete with exit0, after_vram0%, no visible KFD user and no surviving task container.
Target HCU3/gfx938/wave64, gateway77a2848, DTK image locator3ad0ae7192b8, Torch2.11.0, vendor Triton3.6.0.
The per-user gateway does not prove physical exclusivity; other host activity is outside this experiment's control.

## Disposition

No promotion. Retain the qualified per-shape baselines and the observed tiny S8 effect with its noise limits.
The investigation does not support reducing long-row registers merely by lowering S, nor choosing the widest load solely from assembly.
Use the full tensor-to-lane mapping, actual allocation, dynamic work and complete caller measurements together.
No Compiler layout model, Target fact, default dispatcher or framework replacement was changed.
