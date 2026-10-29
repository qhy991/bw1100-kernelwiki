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
- doc-hygon-moe-aiter-backend
date: '2026-10-31'
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

## 社区对照（源码，非本机实测）
MoE 手写后端把 sorting 当成 GEMM 的前置强约束而非独立算子
（见 doc-hygon-moe-aiter-backend）：INT8 强制 CK sorting、W4A8
强制 moe_align、FP8 两者皆可，约束写成 TORCH_CHECK 而非静默回退；
排序后的 token ids 用"低 24 bit token id + 高 8 bit topk id"压缩
编码，GEMM 主循环里 `inline_buffer_load_dword` 预取后展开为 A 行
地址。真实 sorting kernel 在 aiter/CK 侧（Moe 仓库内的
`moe_sorting_fwd` 只是 Python 模拟器，语义一致性未逐行核对）。
本题若走"排序换专门分桶"路线，社区的压缩编码 + 预取展开是现成
参照。
