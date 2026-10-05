---
id: kernel-bw-rope
title: RoPE cos/sin：输入频率、社区分母与 dispatch
type: wiki-kernel
architectures:
- gfx938
tags:
- rope
- runtime-dispatch
- community-baseline
confidence: experimental
sources:
- exp-community-baselines
- exp-register-values
date: '2026-10-05'
description: RoPE cos/sin生成的优化重点是复用公共metadata、选择合适执行路线，并避免额外布局搬运。
kernel_types:
- rope
- embedding
languages:
- python
- triton-rocm
techniques:
- runtime-dispatch
- layout-transform
related:
- technique-existing-layout-maps
- pattern-storage-rebinding
---

RoPE cos/sin生成的优化重点是复用公共metadata、选择合适执行路线，并避免额外布局搬运。

## 优化思想
把计算绑定到调用者提供的frequency/position。若多个实现适合不同公开size，可按shape选择compiled或eager路线，把dispatch费用一并计时。
融合或改写access maps可以省去独立transpose/copy，但需要让后续访问仍对应原元素。

## 代价与条件
compiled路线有初始化/JIT成本，eager路线有较多host和kernel调用；收益取决于实际shape及warm/cold计时范围。频率缓存不能替换外部输入，结果缓存也不能跳过重算。

## 本机范围
Transformers公开size dispatcher有原workload资格；旧handwritten cache是不同分母。
一个小shape的BF16copy证明已有access maps能表达该映射，不等于完整RoPE或任意register permutation已获得收益。
