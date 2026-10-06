---
id: doc-triton-grouped-gemm
title: Triton grouped GEMM program ordering
type: source-doc
architectures: []
tags:
- gemm
- tiling
date: '2026-10-06'
url: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
confidence: source-reported
---

Triton main matrix multiplication tutorial，采集于 2026-10-06。

分组映射 program ID 可在相邻输出 tiles 间复用操作数，减少重复取数；更改的是访问顺序，不是数学 GEMM。
BM/BN/BK、并行线程和 pipeline 决定不同资源代价；原示例给出的 autotune 集合不是 Hygon 最优值。

本轮只收集机制，没有 gfx938 grouped-GEMM 对照。下一步须固定 tile、精度与 MMAC 路径，只改 program ordering，记录 L2/流量及完整调用时间。
大 tile 提高复用也可能增加 registers/LDS；小矩阵没有足够 CTA 时分组可能无益。
