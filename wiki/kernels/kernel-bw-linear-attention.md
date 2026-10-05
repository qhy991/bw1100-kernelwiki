---
id: kernel-bw-linear-attention
title: Chunk gated delta rule：混合 gate 与 FLA component
type: wiki-kernel
architectures:
- gfx938
tags:
- attention
- precision
- fla
- lds
confidence: experimental
sources:
- exp-community-baselines
- exp-night-exclusions
date: '2026-10-05'
description: Chunk gated delta rule 的优化需要先对齐每个阶段的gate语义，再选择state更新的块大小。
kernel_types:
- attention
- reduction
languages:
- triton-rocm
- python
related:
- pattern-precision-not-output-only
- pattern-storage-rebinding
---

Chunk gated delta rule 的优化需要先对齐每个阶段的gate语义，再选择state更新的块大小。

## 计算组织
chunk结构把序列划分为局部计算和跨chunk状态更新。不同site可能使用不同gate；head映射、padding和状态精度都会影响结果。单个同名fused入口不一定表达这个组合。

## 优化思想
复用已优化的组件kernel，按原数学连接各阶段。state的V tile、执行组和pipeline深度共同决定寄存器/LDS footprint；较小tile可减少单CTA资源，但会增加tile数或重复工作。
图重放可以减少host dispatch，若每次把当前输入复制到owned buffer，copy成本也应计入完整调用。

## 本机范围
已有未改动FLA组件的FP32 mixed-gate组合资格，以及限定state映射的profile。它纠正的是早期adapter判断，不表示任意fused FLA入口兼容。未独立确认的搜索分数只保留为候选记录。
