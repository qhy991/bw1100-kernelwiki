---
id: kernel-bw-rmsnorm
title: 残差 RMSNorm：AITER baseline、broadcast 与 host 开销
type: wiki-kernel
architectures:
- gfx938
tags:
- rmsnorm
- aiter
- broadcast
- host-overhead
confidence: experimental
sources:
- exp-community-baselines
- exp-rms-confirmation
- exp-host-entry
date: '2026-10-05'
description: 残差 RMSNorm 的优化重点是减少整行数据的重复读写，并控制归约的线程映射。
kernel_types:
- rmsnorm
- reduction
languages:
- triton-rocm
- python
techniques:
- kernel-fusion
- runtime-guard
related:
- technique-register-scan-broadcast
- pattern-storage-rebinding
- technique-host-entry
---

残差 RMSNorm 的优化重点是减少整行数据的重复读写，并控制归约的线程映射。

## 计算与主要代价
每行先得到残差相加结果，再计算平方和/均值、倒平方根和带权缩放。除了算术，输入、中间残差与输出的HBM往返、kernel启动及归约同步也产生代价。

## 优化思想
- 在同一kernel中复用已加载的残差结果，完成统计量和缩放，减少中间global materialization。
- 使用寄存器broadcast扩展每列权重，不需要先构造一个重复的global权重矩阵。
- 为invalid lanes选择符合归约代数的中和值。带bias的表达式要在bias处理后再次保证padding被中和。
- 用执行组数量调节一行工作的分摊。较多线程可降低单线程持有量，但会增加跨wave归约或同步；较少线程则可能增加寄存器压力。

## 适用边界
保留Task规定的残差舍入和FP32统计链。完整row的归约与跨CTA归约是不同问题，resident算法不自动覆盖任意长row。
对短调用，host分配/检查/dispatch也可能占明显比例，需要同kernel完整调用对照。

## 本机观察
显式broadcast表示及原算法都通过原Task确认。两者分别对社区基线计时，没有直接同次配对，因此不据小比值差异判断broadcast带来额外收益。原始数字和版本见sources。
