---
id: doc-telemetry-sampling-scope
title: Telemetry scope, update delay and sampling intervals must precede short-kernel attribution
type: source-doc
architectures: []
tags: [profiling, local-evidence]
date: '2026-10-08'
url: https://arxiv.org/abs/2604.06056v2
confidence: source-reported
---

Adam McDaniel et al., *Fine-Grained Power and Energy Attribution on AMD GPU/APU-Based Exascale Nodes*, arXiv:2604.06056v2, revised2026-04-09.
The abstract describes sensor-scope, update-rate, delay, filtering and aliasing differences on MI250X/MI300A-based Cray systems.
Its controlled workload methodology motivates validating the observation interface before attributing short-lived device activity.
This source record uses that methodological point; it does not import the paper's energy savings, sensor periods or hardware constants into gfx938.

[ROCm SMI usage documentation](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/how-to/use-python.html)
separates display operations from clock/performance controls and directs users to the installed tool's help for actual options.
The local Hygon tool is hy-smi, not an assumed drop-in AMD SMI interface.
exp-clock-observation-20261008 records its actual version, read-only flags, raw output, query intervals and same-host timestamp comparison.

A timestamp around a CLI invocation bounds the software query, not necessarily the underlying sensor's update instant or averaging window.
An overlapping query can include neighboring work or idle time. It does not automatically provide a kernel-specific state sample.
Read-only operations also have a measurement cost; absence of writes does not establish absence of perturbation.
