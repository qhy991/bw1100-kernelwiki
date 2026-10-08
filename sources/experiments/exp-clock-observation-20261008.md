---
id: exp-clock-observation-20261008
title: Read-only HCU telemetry observes reported states but its query interval exceeds the diagnostic compute blocks
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, profiling, host-overhead, correctness]
confidence: experimental
date: '2026-10-08'
evidence_scope: component-only
evidence_root: archive:wiki-clock-observation-qualified-20261008
artifacts:
- hcu_clock_observation.py
- binding.json
- prepare.json
- prepare.log
- telemetry.jsonl
- worker.jsonl
- worker.log
- observer.log
- device-admission.json
- device-admission-terminal.json
- analyze.py
- analysis.json
- analysis.log
- analyze_centered.py
- analysis-centered.json
- analysis-centered.log
source_commit: 7f402be0
status: completed
limitations:
- No kernel optimization or new speed claim; A1 and A2 replay the same graph
- Reported clock levels are not per-kernel instantaneous effective-frequency measurements
- Query/sensor delay and observer perturbation were not isolated
- This run cannot reconstruct clocks or explain the historical regimes in exp-argmax-bf16-inline-20261008
- One finite input and one shape are diagnostic scope, not replacement qualification for the full BF16 task
---

The logical archive ID is owned by the experiment. Public records omit internal hosts and account paths.
An earlier CPU-only preparation at source426c37cf remains in archive:wiki-clock-observation-20261008.
Before GPU work, successor7f402be0 adds the actual16-byte pointer check required by the frozen kernel's compilation attributes; the predecessor has no GPU admission.

## Observe an unresolved state boundary without changing settings

exp-argmax-bf16-inline-20261008 retained two markedly different absolute-time regimes for the same kernel.
That record contains no concurrent clock trace and cannot establish the physical cause.
This successor qualifies a read-only observation path. It does not append a preferred third timing batch to that experiment.
The upstream measurement-scope rationale is recorded in doc-telemetry-sampling-scope.

The installed tool reports hy-smi1.26.0, build2026-05-29, revision422a6c472751be7fe0be435fd615db9f99c0e180.
Its own help declares display flags for clocks, power, temperature, use, performance level and throttling reason.
The observer invokes only:

```sh
hy-smi -d 3 --showclocks --showpower --showtemp --showuse --showperflevel --showfdr
```

It captures raw stdout/stderr, exit status, UTC and monotonic start/end times, and parsed fields.
The first query must succeed with complete fields before any GPU launch.
Later query failures would remain missing observations; they cannot restart or cancel a worker.
No clock, power, fan, performance-level, reset or service-control command is issued.
Manual mode is already reported before this run. Its earlier origin is unknown.

## Fixed diagnostic workload and existing ownership

The worker imports the frozen uint32 BF16 argmax kernel from sourceb84dedbd and verifies its preceding assembly during CPU preparation.
It reuses the completed numerical/profile qualification from source97733001.
The single diagnostic cell is M4097/N1024 with the existing finite input, exact original BF16 bits, int64 indices and guarded fixed outputs.
There is no new algorithm or candidate; no fresh bottleneck/profile claim is made for this harness.

A captured graph contains eight requested calls. Each timed member submits1,024 replays of that same graph.
There are12 A/A pairs,24 timed blocks, alternating A1/A2 order, and eleven fixed0.25s idle intervals between pairs.
Reset, correctness checks and idle intervals remain outside each timed block.
The requested8,192 calls per member define this diagnostic loop; no new dispatch-count profile was collected.
This long repeated workload has a different interval and cache history from the previous eight-call measurements and is not compared with their absolute latency.

The observer runs on the host and launches the worker only through the existing dtk.sh/HCU gateway, with a90s device limit.
The gateway remains the sole device owner and retains its per-user lock, admission, timeout and release checks.
Sampling covers the gateway process lifetime, including setup and release; the label during does not mean a kernel is continuously active.
Fixed idle periods are part of this short diagnostic interval, not an opportunity to lend out an active worker's allocation.

Job bw-a89642334244 completes exit0 with24 exact-output/input/guard checks, a validated first replay and explicit graph reset.
The gateway reports after_vram0%, no visible KFD user and no surviving task container.
The observer also terminates normally; no monitor remains running.
Target HCU3/gfx938/wave64, image locator3ad0ae7192b8, gateway77a2848, Torch2.11.0.
Physical exclusivity remains unproved.

## Actual query cost and temporal coverage

All96 queries complete and parse successfully. The requested sampling interval is0.25s, not a promise of instantaneous observations.
The query time is measured around the whole CLI process:

| Observation | Minimum | Median | Maximum |
|---|---:|---:|---:|
| CLI query duration, ms | 139.922 | 167.012 | 297.550 |
| Worker timed-block wall duration, ms | 119.045 | 119.129 | 119.457 |
| Worker HIP-event interval, ms | 118.990 | 119.078 | 119.406 |

All24 worker blocks overlap at least one valid query interval.
Zero blocks contain a complete query interval. Twelve queries overlap two neighboring worker blocks.
Thus an overlap does not isolate the observed state to one A/A member, even for approximately119ms blocks.
This measured combined CLI path is unsuitable for assigning an instantaneous clock to microsecond-scale kernels.
The result does not rule out faster or more precisely timestamped vendor interfaces that were not tested.

Host and worker UTC/monotonic pairs are consistent enough for this coarse interval join.
After centering integer offsets before taking medians, their observed median offset difference is10ns; maximum UTC-versus-monotonic duration discrepancy is6.38μs.
Those are software timestamp consistency observations, not a10ns synchronization-accuracy claim or a sensor-timestamp calibration.
The first analysis took medians of epoch-sized integers through floating point and rounded the offset difference to0.
It is preserved as analysis.json; analysis-centered.json owns the corrected comparison. Raw data, timings and coverage counts are unchanged.
No state is interpolated into gaps or assigned solely from a nearest timestamp.

## Reported states and matched A/A observations

Every successful query reports:

| Field | Observed report |
|---|---|
| Performance level | manual |
| sclk | 1300 MHz |
| mclk | 875 MHz |
| fclk | 1300 MHz |
| socclk | 1200 MHz |

Other reported fields vary over the full observer window:

- Average package power:163–281W.
- Average GFX core power:2–96W.
- HCU use:0–90.6%.
- Edge temperature:56–58°C.
- Throttler status:IDLE Active or NONE.

Before and after the worker, the reported performance level and clock values match; use is0% and IDLE Active is present.
These reports do not prove an invariant effective clock for every device cycle or a calibrated instantaneous power value.
They also cannot identify whether or when settings changed before this experiment.

The two A labels execute the same graph. Wall A1/A2 ratios span0.997263–1.000629; event ratios span0.997208–1.000450.
That describes this fixed long-block diagnostic only.
There is no observer-off control, so it does not prove that telemetry collection has zero overhead or that these timings equal an uninstrumented run.
The earlier absolute-time regime change remains unexplained.

## Disposition

No promotion. The read-only CLI path and its raw/parsed/timestamp chain are usable as coarse context with explicit coverage limits.
Do not treat a clock-level string as an instantaneous kernel clock, infer an energy integral from sparse average power, or use current manual mode to rewrite old evidence.
Do not add clock-setting or continuous monitoring behavior to default experiments on the basis of this observation.
No Compiler, Target calibration, measurement default, GPU setting or unrelated service was changed.
