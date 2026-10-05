---
id: kernel-bw-expert-sort
title: 稳定专家分桶：整数精确性与 prefix-sum 范围
type: wiki-kernel
architectures:
- gfx938
tags:
- stable-sort
- int32
- scan
- correctness
confidence: experimental
sources:
- exp-community-baselines
- exp-register-values
- exp-night-exclusions
date: '2026-10-05'
description: 稳定专家分桶可以把通用排序代价换成对专家值域的专门处理，但必须保留原顺序。
kernel_types:
- reduction
- histogram
- moe
languages:
- triton-rocm
- python
related:
- technique-register-scan-broadcast
- pattern-storage-rebinding
---

稳定专家分桶可以把通用排序代价换成对专家值域的专门处理，但必须保留原顺序。

## 优化思想
对expert id稳定排序后，用lower-bound/searchsorted得到各expert分段offset，是一个已有社区路径。另一类方法先计数、做prefix，再scatter；它可能少做比较，但需要处理并发、稳定性和原索引。

当offset只需要固定expert值域的边界时，查询排序后的值可以比再逐段归约更直接。具体收益取决于值域、token数与已有sort成本。

## 整数语义
permutation和offset是离散输出，必须exact。INT32 prefix不应先转FP32，否则超过2^24可能丢信息。resident scan只覆盖一个寄存器域；global carry和跨CTA稳定scatter仍是额外工作。

## 本机范围
已有native sort/search与histogram/prefix composition的具体资格和比较。旧graph候选的storage行为需要单独验证，不从搜索分数推通用推荐。
