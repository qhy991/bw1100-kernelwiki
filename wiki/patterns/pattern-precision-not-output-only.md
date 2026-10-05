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
