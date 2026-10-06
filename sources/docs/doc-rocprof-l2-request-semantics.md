---
id: doc-rocprof-l2-request-semantics
title: L2 request metrics and hit-on-miss interpretation
type: source-doc
architectures: []
tags: [profiling, rocprof, tiling]
date: '2026-10-06'
url: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.1/conceptual/l2-cache.html
confidence: source-reported
---

ROCm Compute Profiler 3.1.0 / docs-6.4.1，2026-10-06 读取。
AMD 的 L2 hit rate 按 cache-line requests 计算，不按 kernel 源代码的 tensor 元素数。
文档描述的 hit-on-miss 会把等待同一 pending line 的后续请求计为 hit，所以高命中率不等于无等待。

该来源解释 AMD 工具口径，不为 gfx938 提供 cache-line size、通道数或峰值。
BW1100 采集需保留本机 derived formula 和 actual kernel 行，不能把外部文档常数直接乘回流量。
比较 group ordering 时固定 kernel、输入、state reset 和 dispatch 顺序；counter 与速度分开采集。
单次 dispatch 的 profile 不等于多次 replay 的缓存状态，不能直接拼接成精确 roofline。
