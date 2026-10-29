---
id: kernel-bw-gateup
title: 双 GEMM＋GELU：保留投影舍入的 tiled fusion
type: wiki-kernel
architectures:
- gfx938
tags:
- gemm
- activation
- fusion
- precision
- execution-groups
confidence: experimental
sources:
- exp-community-baselines
- exp-gateup-fusion
- exp-width-qualification
- doc-hygon-moe-aiter-backend
date: '2026-10-31'
description: 双投影 gate 的融合可以减少中间张量读写，但会同时延长两份累加器的生命周期。
kernel_types:
- gemm
- activation
- custom-fusion
languages:
- triton-rocm
- python
techniques:
- kernel-fusion
- launch-configuration
related:
- technique-rounded-tiled-fusion
- technique-execution-groups
- pattern-fp32-staging
---

双投影 gate 的融合可以减少中间张量读写，但会同时延长两份累加器的生命周期。

## 数据流
本题计算两个投影，gate分支使用GELU-tanh，与up分支相乘。题名中的SwiGLU不替代实际reference语义。两次投影有规定的BF16舍入边界，accumulation及activation使用FP32。

## 优化思想
把两个GEMM放在同一个K循环内，可在局部复用输入tile。两个投影的累加器完成后，保留显式BF16舍入，再在寄存器中计算activation和乘法，最后写出结果。这样可省去gate/up中间global store和reload。

## 代价交换
融合减少访存和launch，但两个accumulator与activation临时值会同时占用寄存器。较大的M/N tile或较少的执行组可能引入scratch访问。分离kernel、缩小tile或改变执行组仍可能更合适。

## 适用条件与观察
中间结果必须private、consumer唯一、tile访问域对应、显式舍入保留。本机已有手工融合和Compiler推导两种来源；来源页的社区相对比值不能单独说明自动pass比手工实现更快。width2/8的资源和性能观察支持继续比较映射，而不是固定一个默认参数。

## 社区对照（源码，非本机实测）
海光 Moe 手写 GEMM 后端（见 doc-hygon-moe-aiter-backend）的 gate+up
路径有一个直接相关的负面记录：**fp16 累加器溢出**（CK fused_moe
同样记录）。本题 accumulation 规定 FP32——社区实现证实这不是保守
要求，fp16 累加在真实 gate+up 尺度上会出数值问题。该页的
BLOCK_MNK 16×128×128 / WARP 16×32×64 / MMA 16×16×32 tile 词汇
是本题 GEMM 侧的社区分块参照（词汇对照见
technique-hcu-layout-contracts）。
