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
date: '2026-10-05'
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
