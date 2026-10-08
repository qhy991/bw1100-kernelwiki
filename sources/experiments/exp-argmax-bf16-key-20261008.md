---
id: exp-argmax-bf16-key-20261008
title: BF16 keys fit one word but installed Torch GPU max does not preserve every selected NaN payload
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, bf16, correctness, reduction, assembly]
confidence: experimental
date: '2026-10-08'
evidence_scope: partial-runtime-triage
evidence_root: archive:wiki-argmax-bf16-diagnostic-20261008
artifacts:
- argmax_bf16_diagnose.py
- prepare.json
- prepare.log
- snapshots.jsonl
- device.log
- device-admission-terminal.json
- analyze.py
- analysis.json
- analysis.log
- installed-headers.txt
- installed-bf16-headers.txt
- installed-bf16-definition.txt
source_commit: 523e3f58
status: completed
limitations:
- Diagnostic correctness and compile observations only; no paired timing or qualified full profile
- A new BF16 task, not a precision reduction of any FP32 task
- Device coverage M63/4097 and N129/1024 only; encoding range is not device qualification
- Source headers suggest a conversion path but the installed Torch binary path was not uniquely traced
---

The archive ID is logical. Internal hosts, user paths and raw runtime headers are not copied into the public wiki.
The frozen candidate/failed-profile archive is archive:wiki-argmax-bf16-key-20261008, source b84dedbd.
It owns argmax_bf16_key_probe.py, binding.json, inputs, key-audit.json, compiled, prepare.log, profile.jsonl, profile.csv, profile.log, profile-admission-terminal.json, audit_machine.py and machine-audit.json.
The diagnostic successor owns the artifact list above and imports the same kernel bodies.

## A distinct BF16 contract

The inputs are BF16 bit patterns. Select the first NaN if present, otherwise the first numeric maximum; positive and negative zero tie.
Return the selected original BF16 bits and an int64 index. Inputs and guards must remain unchanged.
This is a deliberately bit-preserving local contract. It is stronger than a claim that an output merely belongs to the NaN class.
It does not reinterpret an existing FP32 workload as BF16 to obtain a speed result.

The three proposed arms are actual Torch max(dim=1,out=...), a uint64 key and a uint32 key.
Both native arms reload the selected original value from its winning index, so ordering normalization cannot replace the returned payload or zero sign.
Both use identical BF16 reads, int64 outputs, grid, layout and options except key width.
The planned timing boundary uses fixed output buffers and both eager/graph execution; it has not been measured successfully.
No fresh-allocation or end-to-end model claim follows.

## Why 32 bits suffice in the declared encoding domain

For raw BF16 bits r, construct an unsigned16 ordering field h:

- Flip all16 bits for negative numbers, or flip the sign bit for nonnegative numbers.
- Map both zero signs to0x8000.
- Map all NaN patterns to0xffff.

For index i, the compact key is `(h << 16) | (0xffff - i)`.
The uint64 control instead uses `(uint64(h) << 32) | (0xffffffff - i)`.
Unsigned max selects the highest value class and the smallest original index within a tie.

CPU checks enumerate all65,536 BF16 patterns:65,282 numeric patterns and254 NaNs.
The numeric ordering is monotone, zeros share a field, and NaNs outrank all numeric fields.
The smallest valid h is0x007f, so valid compact keys remain above zero padding even when the low field is zero.
The16-bit low field represents indices0..65535, hence nonempty rows require N<=65536 for this encoding.
Index65536 no longer fits the low field. This is an encoding boundary; device programs were compiled and tested only for N129/1024.

Sixteen generated finite/zeros/subnormal/special cases cover33,280 rows.
Both key encodings match the independent FP32-expanded BF16 oracle, and the installed Torch CPU results match all values and indices.
A forward-index tie control fails23,614 rows; interpreting the compact key as signed fails19,692 rows.
These controls reject real ordering errors without changing the accepted inputs.

## Narrower communication does not guarantee fewer registers

Eight native instances compile on vendor Triton3.6.0 for gfx938/wave64.
The following are static assembly observations, not physical allocation or performance measurements:

| N | uint64 key source VGPR | uint32 key source VGPR | DPP sites64→32 | unsigned64 compare sites64→32 |
|---|---:|---:|---:|---:|
| 129 | 16 | 15 | 12→6 | 8→0 |
| 1024 | 29 | 54 | 12→6 | 21→0 |

The compact route uses unsigned32 max sites, including DPP forms; each route retains one readlane site.
Both have zero static DS sites. For N1024 both have four dwordx2 load sites plus the winner ushort reload; N129 has four ushort sites.
The narrower key changes generated dataflow and allocation. This record does not identify the reason for the long-row VGPR increase or a limiting resource.
Full dynamic resource/profile qualification remains incomplete.

## The first device divergence stops timing

Profile job bw-13b536031fff fails at M63/N129, special input, Torch eager.
The prior finite, zero and subnormal checks at that shape had passed for all three methods in eager and graph modes.
The assertion combined value, index and input checks, so its traceback alone did not localize the cause.
The job exits1/not_qualified. Release observations show0% VRAM, no visible KFD user and no remaining task container.
No run or confirm timing batch was started. The partial CSV is retained and never used as a speed score.

Diagnostic source523e3f58 first checks all eight assemblies against the frozen candidate, then captures separate value-bit and index snapshots.
Job bw-a417d1f95690 completes48 comparisons: four shapes × four input patterns × three methods.
These diagnostic calls are ordinary eager invocations; full special-value graph coverage remains unqualified.
All input and guard checks pass. Both native key widths match every value and index on all16 cases.
Torch indices also match every case. Its value bits differ only in the four special-input cases:

| M,N | Torch value-bit mismatch rows | index mismatch rows |
|---|---:|---:|
| 63,129 | 32 | 0 |
| 63,1024 | 23 | 0 |
| 4097,129 | 2116 | 0 |
| 4097,1024 | 1503 | 0 |

All3,674 observed mismatches convert expected0x7f81,0xff81 or0xffc1 to0x7fc0.
The first selected index remains correct. Not every NaN row changes bits, and counts vary with shape; do not generalize this to unconditional canonicalization of every output.
The diagnostic returns exit0/completed and also releases with0% VRAM, no visible KFD user and no remaining container.

Runtime identity: Torch2.11.0, git fb4aa77f70ff922133bad8bbb8649f3d953190b6, HIP6.3.26113, DTK image locator3ad0ae7192b8, gateway77a2848, HCU3/gfx938.
This is an installed-build observation. It is not a claim that every Torch version, GPU or reduction path behaves identically.

## Header evidence is a mechanism lead, not a binary trace

The installed c10 BFloat16 headers forward to torch/headeronly/util/BFloat16.h.
Its float constructor uses a rounding helper whose NaN branch returns0x7fc0; conversion to float follows a separate path.
SharedReduceOps.h passes the value and index through WARP_SHFL_DOWN, and DeviceUtils.cuh exposes a generic HIP shuffle route.
Together these definitions suggest that a value conversion during communication can change payload bits while the independently shuffled index remains correct.
That inference is consistent with the snapshots; this investigation did not trace the exact prebuilt Torch instructions or prove the unique conversion site.
A correct index therefore does not imply an original-bit value result.

## Disposition

No promotion and no speed claim. The exact original-bit contract remains unchanged.
Torch is not a qualified comparator for this stronger BF16 contract across the declared special inputs, despite the CPU match.
This does not establish a general Torch correctness bug: the experiment demands original NaN sign/payload preservation, which ordinary NaN-class equivalence does not require.
A follow-up may continue the same-contract native key-width comparison or investigate the conversion path; it must retain the failed library comparison and declare its own evidence boundary.
