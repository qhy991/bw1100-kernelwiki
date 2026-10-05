---
id: technique-rounded-tiled-fusion
title: 私有 rounded tile 的 Program 融合条件
type: wiki-technique
architectures:
- gfx938
tags:
- fusion
- bf16
- fp16
- precision
- tiling
confidence: experimental
sources:
- exp-gateup-fusion
date: '2026-10-05'
description: 融合private rounded tile的目标是省去producer到consumer之间的global写回和重新读取。
techniques:
- kernel-fusion
- tiling
related:
- kernel-bw-gateup
- pattern-precision-not-output-only
---

融合private rounded tile的目标是省去producer到consumer之间的global写回和重新读取。

## 优化思想
producer完成一个tile后，在寄存器里把显式舍入结果交给pointwise consumer。若consumer逐元素、tile域相同且没有其他观察者，中间global tensor可以删除。多个投影可以一起转发。

## 条件
完整Program应证明中间结果private/sole-consumer。producer loop保持，cast在原位置；maskedCartesian访问域、role/residency承诺一致。额外consumer、不同view/transpose、consumer归约或effects需要另一个变换域。

## 代价
融合减少访存与launch，但延长register生命周期，并可能阻碍不同阶段各自选择tile。FP32算术materialization删除还可能改变指令收缩；identity cast不是可靠舍入屏障。

## 本机范围
四个small-M dual-GEMM/GELU可由pass推导，并保留原任务结果。现有手工融合并非新的收益分母，自动推导和kernel性能是两个独立观察。
