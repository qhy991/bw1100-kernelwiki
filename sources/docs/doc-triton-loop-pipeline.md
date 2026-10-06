---
id: doc-triton-loop-pipeline
title: Triton loop pipeline attributes and actual lowering
type: source-doc
architectures: []
tags: [triton, gemm, vgpr, lds]
date: '2026-10-06'
url: https://triton-lang.org/main/python-api/generated/triton.language.range.html
confidence: source-reported
---

Triton main tl.range API文档，2026-10-06读取。
loop num_stages尝试重叠循环中的loads，与kernel num_stages重点处理喂给dot的loads存在区别。
文档还区分unroll、accumulator multi-buffer与warp specialization；不能把这些开关统称同一机制。

本轮先改变kernel num_stages，未使用loop属性或warp specialization。
对gfx938应查看本机vendor lowering：请求stage数不证明实际流水深度；无效果、拒绝和
更高资源代价都应保留。此来源没有给出Hygon最优stage数，也不能授予上游其他target专属能力。
