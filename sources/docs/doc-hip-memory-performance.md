---
id: doc-hip-memory-performance
title: HIP coalescing and resource tradeoffs
type: source-doc
architectures: []
tags:
- tiling
- lds
- occupancy-tuning
date: '2026-10-06'
url: https://rocm.docs.amd.com/projects/HIP/en/develop/understand/performance_optimization.html
confidence: source-reported
---

HIP develop 文档（页面显示 7.17.0），采集于 2026-10-06。

相邻线程的相邻 global 地址有利于合并事务。LDS 的访问需要分析 bank，不能用 global coalescing 代替。
增加线程、寄存器或 shared tile 会改变资源占用；occupancy 本身不是最终速度。

迁移到 gfx938 时先写出 lane 到地址的函数，检查实际 vector load/store 和 LDS 指令，再实测。
此页没有给出 Hygon 的 memory transaction 大小、带宽峰值或 residency 常数。
