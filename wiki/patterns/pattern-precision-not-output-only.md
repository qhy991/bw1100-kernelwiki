---
id: pattern-precision-not-output-only
title: 输出 matched ratio 通过不授权降低中间精度
type: wiki-pattern
architectures:
- gfx938
tags:
- precision
- fp32
- bf16
- correctness
- negative-result
confidence: experimental
sources:
- exp-gemm-view-precision-20261006
- doc-pytorch-numerical-accuracy
- doc-triton-dot-precision
- exp-community-baselines
- exp-night-exclusions
date: '2026-10-05'
description: 最早问reference哪个stage要求FP32、在哪里有BF16/整数舍入，而不是先看输出分数。
symptoms:
- precision-contract-failure
- output-only-pass
- rounding-boundary-lost
related:
- kernel-bw-moe-fp32
- kernel-bw-vision-attention
- technique-rounded-tiled-fusion
---

最早问reference哪个stage要求FP32、在哪里有BF16/整数舍入，而不是先看输出分数。

strictFP32 MoE候选即使160输出通过，BF16 MMA仍违背Task。Ragged vision score与probability舍入也不能略去。
GateUp有两projection BF16边界，RMS variance用FP32，backward十个输出及norm reduction保留。
integer permutation/offset比较exact，float32丢失>2^24整数精度。

Task拥有语义/精度规则，Compiler只按指令/类型/effects执行，不能把某个benchmark政策灌进generic IR。
保留失败source、dtype/route与原oracle；不放宽tolerance或把native結果标Cake。
之后source/wrapper/precision审计、caller与独立paired/A-A全部完成才能晋升。

## 原生GEMM的分布边界

exp-gemm-view-precision-20261006从dyadic扩到随机、抵消、小幅值和特殊值。view等价性全部通过，
但2^24+1-2^24的用例得到0而FP64为1；FP32 accumulator不保证最终正确舍入的实数和。
不能为这个结果临时放宽Task容差，也不能将正常舍入敏感性直接升级为Compiler缺陷。

本路径保留了所测FP16 subnormal，反驳把其他AMD型号的FTZ说明无条件转给Hygon。
数值能力要绑定dtype、opcode、数据分布和runtime；本轮不是所有denorm或NaN payload的资格。
