---
id: kernel-bw-vision-attention
title: Ragged vision attention：不要抹掉两次 BF16 舍入
type: wiki-kernel
architectures:
- gfx938
tags:
- attention
- bf16
- precision
confidence: experimental
sources:
- exp-community-baselines
date: '2026-10-05'
description: Ragged attention 可以优化分段调度和中间矩阵搬运，但融合必须保留原舍入链。
kernel_types:
- attention
- flash-attention
languages:
- python
- triton-rocm
related:
- pattern-precision-not-output-only
---

Ragged attention 可以优化分段调度和中间矩阵搬运，但融合必须保留原舍入链。

## 优化思想
variable-length布局用cu_seqlens定位每段。局部分块可减少无效padding；fused attention通常试图避免写完整score/probability矩阵，并在线更新softmax统计。

## 数值条件
本题先把scaled score舍入为BF16，再FP32softmax，probability再BF16后做PV。保留更高精度的在线score看似更精确，却会改变这个reference的计算。改变语义与优化原问题需要分开。

## 本机范围
当前社区组合materialize BMM/softmax保留两处舍入。在线路线的差异已记录。未来优化应先明确是否可在融合内部保留这些边界，再量完整ragged程序与搬运，不能只量一个launch。
