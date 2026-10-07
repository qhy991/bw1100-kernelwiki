---
id: doc-waves-per-eu-hint
title: Waves-per-EU is a compiler resource hint, not observed residency
type: source-doc
architectures: []
tags: [triton, occupancy-tuning, vgpr, register-spilling, correctness]
confidence: source-reported
date: '2026-10-07'
url: https://llvm.org/docs/AMDGPUUsage.html
---

LLVM AMDGPUUsage将amdgpu-waves-per-eu描述为最低/最高waves的优化提示；后端可能无法满足，
与workgroup限制冲突时后者优先。它不是观测到的实际驻留wave数。

[ROCm Triton优化文档](https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html)
将waves_per_eu用于提示编译器控制VGPR，以争取目标occupancy；相应AMD资源示例不作为gfx938常数。
应检查提示是否进入LLVM、最终指令与资源是否改变、是否spill，再比较完整调用。

exp-waves-hint-20261007在本机分开记录提示、相同代码过滤、runtime资源、HIP预测、真实profile和计时。
不把降低VGPR、metadata接受选项或HIP预测直接提升为实际occupancy/性能保证。
