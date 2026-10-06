---
id: technique-grouped-program-order
title: Grouped program ordering：先改变复用距离，再测缓存收益
type: wiki-technique
architectures:
- gfx938
tags:
- gemm
- tiling
- runtime-dispatch
confidence: inferred
sources:
- doc-triton-grouped-gemm
- doc-hip-memory-performance
kernel_types:
- gemm
reproducibility: concept
---

## 可验证假设

相邻输出 tiles 访问同一 A 或 B panel 时，缩短它们的调度间距可能增加缓存复用。
对 GEMM 保持 BM/BN/BK、warp 数、精度、operand mapping 和实际指令相同，仅改变
program_id 到 (tile_m,tile_n) 的映射。最后一个不完整 group 需使用实际 group 大小。

## 代价与反例

调度次序不是执行次序保证。小 grid、已经命中缓存的 panel、不同 M/N/K 比例都可能没有收益。
更大 tile 会同时改变 registers/LDS，不应与 ordering 一起改后声称因果来自 L2。
收集同一 kernel 的 fetch/write 与缓存计数，再用无 profiler 的完整调用作对照。

目前只有上游来源；本轮未在 gfx938 测此项，不能填入经验速度表。
agent 的下一步应先找到已有 Cake 具体映射能表达的合法候选；若不足，保留完整 Schedule 和
当前拒绝诊断，再决定是否形成 Finding。此机制不要求引入通用 layout 代数。
