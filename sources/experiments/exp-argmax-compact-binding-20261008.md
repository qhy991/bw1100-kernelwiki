---
id: exp-argmax-compact-binding-20261008
title: A checked single-view argmax binder reduces caller cost without caching input storage
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, host-overhead, fp32, correctness, paired-timing, profiling, runtime-guard]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: archive:wiki-argmax-compact-qualified-20261008
artifacts:
- argmax_compact_binding_probe.py
- binding.json
- cpu-audit.json
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
source_commit: 1f990cd9
compiler: vendor Triton3.6.0 Gluon; same peeled kernel from cc3cf41c; original binder from ec0d1a50
dtype: original FP32 selected bits and int64 first indices with fresh retained outputs
shape: M63/4097 crossed with N129/1024 and pointer offsets0/1; finite/zeros/subnormal/special
baseline: actual Torch default max and original checked per-call view-plus-slice binder
measurement: three direct comparisons, two six-round ABA/BAB batches with eight retained outputs per block; separate marked instruction profile
limitations:
- Fixed contiguous FP32 shapes and pointer-mod16 0/4 only; no arbitrary stride, autograd, concurrent metadata mutation or multistream qualification
- Input allocation and storage replacement outside timing; all binding checks and view construction inside native calls
- Warm resident inputs and allocator; old-result destruction excluded; physical exclusivity and full cache eviction unproved
- Pointer-query reuse and one-view construction change together; their separate contributions are not identified
status: completed
---

The archive ID is a logical locator retained by the experiment owner. Public records omit internal host and account paths.

## Reduce metadata work while preserving the same checks

exp-argmax-rebind-20261008 measured the cost of checking and rebuilding an anchor on every call.
This successor compares that original binder with a compact binder and with actual Torch max.
Both native arms bind the current input on each call. Neither caches a data anchor across calls.
The frozen peeled kernel, output allocation, exact oracles and old-result checks are unchanged.

The original binder calls data_ptr four times and constructs `x.view(-1)[shift:]`.
The compact binder reads the input pointer once and the anchor pointer once.
It uses one view:

```python
anchor = x.as_strided((M*N-shift,), (1,), x.storage_offset()+shift)
```

It retains dtype, shape, contiguity, allowed pointer remainder, anchor alignment and byte-delta checks.
The dimensions are the same fixed supported dimensions checked against the input.
The view begins at the current storage offset plus shift and ends at the original logical input end.
It does not expose extra elements or copy input data.

PyTorch2.11 [as_strided](https://docs.pytorch.org/docs/2.11/generated/torch.as_strided.html)
uses a storage-relative explicit offset and requires the view to remain in storage.
This experiment is a bounded use of that API, not a general recommendation to replace view operations on other layouts or backends.

## Preserve the incorrect comparison predecessor

Source1b2cf72d in archive:wiki-argmax-compact-binding-20261008 passed its CPU binder checks and ran profile bw-6794cc59b511.
A substring replacement also changed prebound_alloc to pcompact_alloc.
The actual block still executed the old prebound path and old binder; it did not call bind_compact_anchor.
The intended comparison therefore failed even though device outputs and CSV structure passed.
A post-profile check found zero pairs under the intended baseline name. No timing batch was started.

comparison-disposition.json records invalid-comparison and the successor. Original files and the completed release receipt remain intact.
Source1f990cd9 corrects the exact method names and both call branches in a new directory.
A source audit confirms the original binder and compact binder each occur in the actual block.
All results below belong to this successor. The predecessor supplies no performance claim.

One successor SSH attempt timed out before authentication. A read-only check found no pmc, log or admission file before submission was retried.
No accepted GPU job was restarted because of an observation timeout.

## CPU, machine and device evidence

CPU qualification checks32 offset/pattern inputs on changed storage,32 wrong-origin controls and16 matching rejection results.
It compares both anchors' pointers, storage offsets, complete bits and recovered input, while preserving guards.
Eight generated assemblies match the frozen peeled predecessor.

Valid profile bw-f61f16146062 contains1,152 target dispatches across144 marked blocks,40 storage refreshes and96 correctness blocks.
Each scope has exactly eight reductions and no extra copy/helper kernel.
All32 native case-pattern pairs have equal measured Wavefronts, SQ_INSTS_VALU, VALUInsts and LDSInsts.
This observation covers the selected counters; it does not establish every device metric or a cache cause.

Run bw-c6d93005fca0 and confirm bw-0b919654c631 each check72 storage refreshes,96 correctness blocks and1,728 timing blocks.
Across both batches,1,536 fresh-check output pairs and27,648 timed output pairs pass exact FP32-bit/int64-index checks.
All eight results and prior retained result blocks are verified. The same per-shape Tensor object is reused across storage and offset changes.

Target HCU3/gfx938/wave64; gateway77a2848; DTK image locator3ad0ae7192b8; Torch2.11.0 and vendor Triton3.6.0.
The invalid predecessor profile and all three successor GPU admissions completed with exit0, after_vram0%, no visible KFD user and no surviving task container.

## Complete caller measurements

The direct pairs are original/compact, Torch/original and Torch/compact.
Both native intervals include all metadata checks, view construction, fresh output allocation and eight completed calls.
Input generation/storage replacement, validation and prior-result destruction stay outside timing.
The protocol retains six ABA/BAB rounds, per-cell A/A, initialized HIP events, synchronized64MiB resets and a reversed confirmation batch.

Finite input confirmation results, in μs per call. Each row is its own pair.

| M,N,offset | comparison | baseline | candidate | paired baseline/candidate |
|---|---|---:|---:|---:|
| 63,129,0 | original / compact | 37.53513 | 32.65663 | 1.1518 |
| 63,129,0 | Torch / compact | 19.64125 | 32.50050 | 0.6060 |
| 4097,129,1 | original / compact | 37.23525 | 32.43788 | 1.1534 |
| 4097,129,1 | Torch / compact | 22.89613 | 32.60163 | 0.7030 |
| 4097,1024,0 | original / compact | 37.57138 | 32.39163 | 1.1558 |
| 4097,1024,0 | Torch / compact | 40.06488 | 32.74425 | 1.2180 |
| 4097,1024,1 | original / compact | 37.33138 | 32.43550 | 1.1556 |
| 4097,1024,1 | Torch / original | 43.35600 | 37.51388 | 1.1565 |
| 4097,1024,1 | Torch / compact | 43.43225 | 32.41550 | 1.3390 |

Across32 case-pattern cells, original/compact wall ratios span1.140–1.174 in run and1.139–1.178 in confirm.
Per-call submit medians fall by4.910–5.579μs and4.677–5.316μs respectively.
The combined host rewrite has a measured benefit; these data do not identify a standalone cost for either removed view or pointer query.

For largeN1024, Torch/compact spans1.185–1.221 at offset0 and1.297–1.309 at offset1 in run.
Confirm spans1.207–1.231 and1.321–1.339 respectively.
SmallM ratios remain about0.585–0.606; largeN129 remains about0.683–0.711. Improving the native wrapper does not make it a universal Torch replacement.
No ratio is composed by multiplying different pairs or comparing absolute times from previous rounds.

All wall A/A samples remain: run range0.938–1.145; confirm0.483–1.289.
The0.483 control occurs in the small63×129/offset0 finite Torch/compact cell.
For finite large1024 Torch/compact, confirm A/A maxima are1.0925 at offset0 and1.0767 at offset1.
No outlier is deleted and no extra measurement round is added after inspecting these results.

## Disposition

No promotion. The checked single-view route is a bounded candidate with lower caller cost and verified current-storage behavior.
The result supports simplifying host metadata work before adding a persistent input cache.
It does not qualify arbitrary metadata, concurrent mutation, stream lifetimes or a default library replacement.
The incorrect comparison remains evidence of why output correctness and a valid trace alone cannot prove that the intended candidate ran.
