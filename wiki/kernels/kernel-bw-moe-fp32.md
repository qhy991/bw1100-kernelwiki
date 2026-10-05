---
id: kernel-bw-moe-fp32
title: 严格 FP32 MoE：混合输入与专家链精度
type: wiki-kernel
architectures:
- gfx938
tags:
- moe
- fp32
- precision
- mmac
- valid-extent
confidence: experimental
sources:
- exp-community-baselines
- exp-fp32-staging
- exp-width-qualification
- exp-night-exclusions
date: '2026-10-05'
description: 稀疏 MoE 的优化要同时处理路由布局、专家矩阵计算和combine的内存路径。
kernel_types:
- moe
- grouped-gemm
- gemm
languages:
- triton-rocm
- python
techniques:
- launch-configuration
- masking
related:
- pattern-precision-not-output-only
- pattern-fp32-staging
- technique-execution-groups
---

稀疏 MoE 的优化要同时处理路由布局、专家矩阵计算和combine的内存路径。

## 主要代价
token经top-k路由后形成不同长度的expert段。逐expert调用会产生小矩阵和较多dispatch；block padding又可能产生无效行。专家计算的中间结果还要按原token顺序组合。

## 优化思想
- 把路由后的token分组为block-aligned布局，让同一expert的行共享权重访问和GEMM调用。
- gate/up在同一阶段计算，并在FP32累加器上完成SiLU与乘法，减少一个中间buffer。
- down的epilogue可乘路由权重，combine再按inverse index聚合top-k结果。
- 用运行时有效长度限制padding的load/store。空block可以跳过，但不能遗漏合法行。
- 选择tile、执行组和unroll时一起检查register/shared占用。改变其中一个值可能把成本转移到另一个资源。

## 数值与范围
本Task保留FP32专家链和combine，最后一次BF16舍入。输入BF16的上转换不能被省略为BF16 GEMM。运行时prefix mask是逻辑有效域，width改写可以保留其定义，但新二进制仍需原Task检查。

## 本机观察
FP32 matrix ISA和dynamic LDS都已有证据；资源较大并不说明缺少dot primitive。源码、dtype和profile细节集中在sources，后续假设是同精度下调整映射能否改善完整调用。
