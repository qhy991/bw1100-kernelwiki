---
id: kernel-bw-training-backward
title: GQA / decoder backward：训练原语与多输出合同
type: wiki-kernel
architectures:
- gfx938
tags:
- backward
- attention
- fp32
- community-baseline
confidence: experimental
sources:
- exp-community-baselines
date: '2026-10-05'
description: 训练backward的优化可以复用native梯度原语，并融合elementwise尾部，但必须保留所有梯度输出。
kernel_types:
- attention
- gemm
- normalization
languages:
- python
- triton-rocm
related:
- pattern-precision-not-output-only
---

训练backward的优化可以复用native梯度原语，并融合elementwise尾部，但必须保留所有梯度输出。

## 优化思想
ATen的softmax、dropout、SiLU backward处理常见梯度链，vendor GEMM负责较大矩阵计算。适配GQA布局时，要在原head映射上归约重复的梯度。
把中间elementwise与相邻矩阵epilogue组合，可能减少临时tensor与launch；共享中间值也可能降低重复计算。

## 条件与代价
梯度归约顺序和FP32 norm链影响数值，十个decoder梯度不可只抽一个评价。多个公开输出限制可删除的中间结果；融合还可能增加live values与内存。

## 本机范围
社区primitive composition有完整原Task资格。进一步融合的收益仍需全输出与完整程序计时，不从一个梯度kernel或标量loss推断。
