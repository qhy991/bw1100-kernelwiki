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
- doc-hygon-lightop-fused-ops
date: '2026-10-31'
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

## 社区对照（源码，非本机实测）
海光 lightop 库的 RoPE 全部以**融合形态**存在（见
doc-hygon-lightop-fused-ops）：`fuse_rmsnorm_rope_qkv` /
`fuse_rmsnorm_rope_quant_qkv` / `fuse_gammarmsnorm_rope_qkv` 等，
QKV 切分 + q/k RMSNorm + RoPE + KV store 量化一个 kernel；单独的
裸 RoPE kernel 不在社区提供面上。这佐证本题的优化方向：RoPE 单
算子收益有限，社区把它当 epilogue 融进邻居算子。注意 lightop 的
fused writer 曾用已废弃的 split-K/V cache ABI，现版本保留计算融合、
写入走原生 HIPC op——KV 写入 ABI 必须与本机 vllm-plugin 绑定核对。
