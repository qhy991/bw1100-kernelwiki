---
id: exp-argmax-anchor-cpu-20261008
title: CPU preflight shows why an argmax data anchor must follow current input storage
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, runtime-guard, fp32]
confidence: experimental
date: '2026-10-08'
evidence_scope: protocol-only
evidence_root: archive:wiki-argmax-anchor-cpu-20261008
artifacts: [argmax_anchor_binding_probe.py, cpu-check.json, cpu.log, cpu-via-bash.log]
source_commit: ec0d1a50
status: completed
limitations:
- CPU metadata and input-bit address checks only; no GPU dispatch or argmax output qualification
- No timing or profiler evidence; the cost of per-call rebinding is unknown
- Fixed shapes and contiguous FP32 pointer-mod16 0/4 only; no cache introduced
---

The archive ID is a logical locator retained by the experiment owner. Host and account paths are omitted from this public record.

## Question and source boundary

exp-argmax-peel-20261008 binds an aligned interior anchor during fixed-input setup.
That anchor aliases data. It is not an output-allocation metadata template.
This CPU preflight asks whether the same Tensor object can move to new storage while an old anchor still reads the old storage.
It does not change the preceding device experiment or extend its speed claim.

PyTorch 2.11 documents [set_](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.set_.html)
as replacing storage, size and strides. An explicit [as_strided storage_offset](https://docs.pytorch.org/docs/2.11/generated/torch.as_strided.html)
refers to the underlying storage. It is not an offset relative to the current view.
The installed build is Torch 2.11.0, git fb4aa77f70ff922133bad8bbb8649f3d953190b6, HIP 6.3.26113.
The existing DTK image locator is 3ad0ae7192b8; the repository CPU entry hides all GPUs and uses runc.

## Retained input and observed checks

The probe reads the 16 frozen FP32 bit-pattern inputs from exp-argmax-fp-key-20261008.
Shapes remain M63/4097 × N129/1024. Each shape tests pointer offsets 0/1 against finite, zeros, subnormal and special inputs.
No oracle or data generator is changed. This stage compares input bits and addresses, not argmax outputs.

For each shape, retain an anchor to the initial finite input. Repeatedly call set_ on the same Tensor object with a new guarded allocation.
Bind a fresh anchor from the current view using `x.view(-1)[shift:]`, where `shift=(-offset)&3`.
Check the actual pointer remainder, byte delta, complete recovered input bits and unchanged guards.
A separate fresh Tensor view of the same current storage must produce the same anchor address.

Observed in cpu-check.json:

- All 32 input cases preserve the current input bits and guards.
- All 32 cases retain the Tensor object identity while changing storage.
- All 32 old anchors still expose their original finite input.
- In the 24 non-finite-pattern transitions, stale anchors differ at 8,127–4,195,328 element positions.
- All 32 explicit-zero-storage-offset controls read incorrect input bits.
- Sixteen rejection checks cover noncontiguous layout, wrong dtype, wrong shape and unsupported pointer remainder.

The repeated finite pattern intentionally remains a control: different storage containing identical bits cannot expose stale data by value comparison alone.
The protocol therefore checks storage relationships and changed data separately.
The old storage stays alive through its retained anchor; this is stale-but-valid data, not a dangling-pointer experiment.

## Execution and next acceptance boundary

The first command could not execute scripts/dtk.sh because it lacks an executable bit; cpu.log retains exit126.
The successor invocation used `bash scripts/dtk.sh cpu` without changing file permissions or the environment.
It completed with exit0 and wrote cpu-check.json. No HCU admission or GPU kernel was launched.

The next device successor must compare the frozen prebound kernel against per-call rebinding and the actual Torch caller.
It must check changed input storage, both pointer remainders, all original argmax outputs, retained old results, and full-call timing including rebinding.
Profile must establish whether the wrapper adds any device work. Only then can a new speed claim be assessed.

No promotion. This record supports a caller-boundary lesson; it does not qualify a new GPU adapter or modify the Compiler.


Device successor: exp-argmax-rebind-20261008 completed the frozen-kernel profile, storage/output checks and paired caller measurements. This CPU record retains its original scope.
