---
id: doc-amd-wave-builtins
title: AMD wave builtin strategies and DPP
type: source-doc
architectures: []
tags:
- wave64
- reduction
- assembly
date: '2026-10-06'
url: https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/cross-arch/wavefront-builtins.html
confidence: source-reported
---

AMD ROCm Optimization Guide 1.0.0，采集于 2026-10-06。

wave reduce 可由显式 shuffle、DPP 或 compiler builtin 表达；上游 builtin 支持与目标有关。
源代码一样的 shuffle 未必产生同一条 ISA，所以要检查自己的 lowering。

本轮选用 DTK 已有非 sync shuffle 接口，并保留 width32 与 width64 的完整 block 合并。
尚未测试 gfx938 DPP 手写版本或新版 wave_reduce builtin；这两项仅为后续候选。
原文 AMD 性能陈述不转为 BW1100 速度承诺。
