---
id: exp-clock-sampler-effect-20261008
title: Sampler on/off cohort is not qualified after termination and container device-access failures
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, profiling, host-overhead, correctness]
confidence: experimental
date: '2026-10-08'
evidence_scope: component-only
evidence_root: archive:wiki-clock-sampler-effect-20261008
artifacts:
- plan.json
- preparation-path-receipt.json
- summarize.py
- summarize_dispositions.py
- disposition-summary.json
- disposition-summary.log
source_commit: cf6eb5c0
status: not-qualified
limitations:
- Only one of four planned workers produced timing records; sampler overhead is unknown
- Exit143 does not identify the termination sender or establish a timeout
- Host telemetry and CPU compilation do not prove container device authorization
- No replacement runs, missing-data imputation or comparison to the prior cohort
- Physical exclusivity and historical clock-state causes remain unproved
---

This is a partial runtime-triage record, not an accepted performance comparison.
Logical archive IDs omit private host and account paths. Raw records remain with their experiment owner.
The initial derived-evidence transfer was interrupted before SSH authentication. On2026-10-09, SSH recovered and a create-only transfer archived the disposition files and off-a analysis remotely. The original four terminal outcomes remain unchanged.

## Frozen comparison and preparation

The unresolved question from exp-clock-observation-20261008 is whether read-only hy-smi sampling perturbs the worker.
Before GPU execution, plan.json fixes off-a, on-a, on-b, off-b: two fresh processes per condition, all four required.
Source cf6eb5c0 adds only the observer on/off control. The worker, runtime helper and query parser definitions are AST-equal to source7f402be0.
Both conditions retain before/after queries; on adds queries during the gateway process lifetime, while off only polls its child at0.25s.
No clock, power, performance mode or service setting is changed.

Each worker retains the frozen BF16 key32 kernel, M4097/N1024 finite input, exact original BF16 bits/int64 indices and guarded outputs.
A graph contains eight requested calls; each block submits1,024 replays. Twelve A/A pairs yield24 blocks per process.
Reset, output/input/guard checks and eleven0.25s inter-pair idle intervals remain outside timed blocks.
The source, numerical/profile qualification and limitations are inherited from the previous diagnostic; there is no new kernel candidate.
Target HCU3/gfx938/wave64, image locator3ad0ae7192b8, gateway77a2848, Torch2.11.0, HIP6.3.26113 and vendor Triton3.6 are bound by the preparation records.
The gateway declares90s outer and60s inner limits and remains the only GPU owner.

The initial source bundle was extracted one directory above the planned destination before preparation or GPU work.
Inspection found only the newly created source/plan files and no destination conflicts. They were moved intact to the planned location.
preparation-path-receipt.json preserves that correction. All four CPU preparations then passed with identical worker bindings.
An SSH interruption was resolved before the fixed sequence began; no worker was duplicated because an observer disconnected.

## All four terminal outcomes

| Member | Job | Exit | Valid timing blocks | Queries / during | Terminal observation |
|---|---|---:|---:|---:|---|
| off-a | bw-7856b9e853e8 | 0 | 24 | 2 / 0 | Worker completes and graph is released |
| on-a | bw-f953ba874bab | 143 | 0 | 42 / 40 | Terminates without a worker record; cause unknown |
| on-b | bw-143ce2396086 | 1 | 0 | 27 / 25 | Creator lookup fails, no valid DCU, then zero active drivers |
| off-b | bw-4b16b443d683 | 1 | 0 | 2 / 0 | Same early runtime/device-access failure |

All73 host queries succeed. All four gateway receipts report0% post-run VRAM, no visible KFD user and no surviving task container.
The observer also ends in every member. Release is verified independently of numerical or performance acceptance.
There are24 recorded blocks out of96 planned, one valid worker out of four and no full-cohort estimate.
Missing worker files remain absent; no empty timing records or substituted samples are manufactured.

on-a lasts approximately10.60s from observer launch to completion, while its gateway timestamps span approximately9.96s.
Its configured inner timeout is60s, so elapsed time and exit143 alone do not establish timeout as the cause.
No worker traceback identifies a sender. A bounded Docker-event lookup returned no records; that absence does not prove that no termination signal occurred.

For on-b and off-b, the earliest visible divergence is the runtime creator-identity lookup warning, followed by no valid DCUs.
Triton then raises `RuntimeError: 0 active drivers ([]). There should only be one.` at `get_current_target()` before tensor allocation or worker-record initialization.
This ordering does not establish a unique underlying platform defect, but it prevents treating the terminal error as a kernel correctness failure or evidence that the compiler backend is absent.
A read-only host check found USER and LOGNAME present and matching the effective user; the runtime creator mapping is a separate unresolved boundary.
The identity, permissions, runtime and target were not modified to make admission pass.

## Evidence locations and acceptance

The common archive owns the fixed plan, preparation correction and disposition analysis.
Each member has its own logical root:

- archive:wiki-clock-effect-off-a-20261008
- archive:wiki-clock-effect-on-a-20261008
- archive:wiki-clock-effect-on-b-20261008
- archive:wiki-clock-effect-off-b-20261008

Each retains hcu_clock_observation.py, binding.json, prepare.json, prepare.log, telemetry.jsonl, worker.log, observer.log, device-admission.json and device-admission-terminal.json.
Only off-a has worker.jsonl and the local derived analyze_centered.py, analysis-centered.json and analysis-centered.log.
The strict planned summarize.py is preserved; summarize_dispositions.py records actual outcomes without relaxing complete-cohort acceptance.

After on-a failed, the remaining predeclared members ran once after release checks; none replaced on-a.
Both sampling conditions eventually encountered the device-access failure. That does not establish that sampling caused either failure or that sampling is harmless.
Do not pair off-a with the preceding round's on run, promote the surviving subset, or interpret missing results as zero overhead.
Further device work requires genuine runtime recovery and an identified successor, not replacement of these four immutable attempts.

No promotion. Sampler overhead and the earlier absolute-time regime change remain unknown.
No Compiler, Target, benchmark default or device setting was changed.


## Separate access check after SSH recovery, 2026-10-09

A new runtime-access-only attempt is retained under archive:wiki-runtime-recovery-20261009; it is not a replacement member of this cohort.
Its plan.json, probe.log, device-admission.json and device-admission-terminal.json retain the complete command and outcome.
The existing gateway77a2848, same image locator3ad0ae7192b8 and physical HCU3 are used with a60s outer/40s inner timeout.
The intended checks are exact gfx938/wave64 selection, a small int64 device sum and gateway release. No timing or bottleneck claim is planned, so no optimization profile is required for this access check.

Job bw-9a14be04be00 exits1. The runtime first reports no valid DCUs, then Triton raises zero active drivers at get_current_target.
The tensor allocation and sum assertion are not reached. This log does not repeat the earlier creator-lookup warning; its absence does not prove that account/device authorization is now valid or establish the unique cause.
The terminal receipt confirms after_vram0%, no visible KFD user and no surviving task container.
SSH, archive transfer and server wiki validation now succeed, while container device access remains unqualified. No user identity, runtime, permissions or target was changed.

The server wiki was fast-forwarded to the published round78 overview and validated191 pages with zero errors and zero warnings.
This follow-up does not change the cohort's missing samples, restore its qualification, or provide a sampler-overhead estimate.


After the operator reported device authorization restored, a separate identical access check ran once in archive:wiki-runtime-recovery-authorized-20261009.
Job bw-2957f9e8f3f3 also exits1 before tensor allocation, with no valid DCUs and zero active drivers; its release checks pass.
The original failed attempt is preserved. Neither check is a sampler comparison or a kernel test.

Read-only inspection then finds that the installed docker executable resolves to dcu-docker-hook.
The gateway requests runtime dtk and passes HIP_VISIBLE_DEVICES for its selected card, but supplies neither DTK_VISIBLE_DEVICES nor DCU_VISIBLE_DEVICES.
The selected image and inspected host process have no defaults for these visibility variables.
Installed dtk-ctk reports2.1.3 and dcu-ctk reports2.1.4. Binary strings associate the invalid-DCU message with dcu-container-runtime and include DCU_VISIBLE_DEVICES in the docker hook.
These observations locate an additional platform layer; embedded strings alone do not establish supported argument semantics or prove that an environment-variable change would fix the failure.
The installed help forwards ordinary Docker options without explaining the platform-specific selection policy.
The read-only diagnostic script and selected output are inspect_runtime.py and runtime-inspection.json under the authorized-check archive.
The platform's supported single-card launch contract remains requested. No hook bypass, creator override, runtime replacement or shared gateway edit was attempted.
