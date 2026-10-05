---
id: pattern-fp32-staging
title: FP32 dot 很慢或 native LDS0：先核对实际 lowering
type: wiki-pattern
architectures:
- gfx938
tags:
- fp32
- mmac
- lds
- vgpr
- profiling
- scratch
- register-spilling
confidence: experimental
sources:
- exp-fp32-staging
- exp-width-qualification
date: '2026-10-05'
description: FP32 GEMM的优化应区分算术指令选择、tile数据搬运和register/shared分配。
symptoms:
- slow-fp32-dot
- fixed-lds-zero
- register-spill
related:
- kernel-bw-moe-fp32
- technique-execution-groups
- technique-profile-gfx938
---

FP32 GEMM的优化应区分算术指令选择、tile数据搬运和register/shared分配。

## 分析思想
沿source、TTGIR、native、实际dispatch核对同一个kernel。若已使用matrix ISA，就把问题转向映射和资源，而不是按慢直接推断scalar fallback。
LDS可由SDK动态分配；native的fixedLDS0没有覆盖这部分。逻辑register估计也不能代替实际allocation。

## 下一步假设
在同IEEE精度与原Workload下改变tile、执行组或unroll/pipeline，观察资源是否变化，以及完整调用是否改善。减少register可能增加LDS，减少LDS可能增加重复load，二者需一起量。

## 本机范围
MoE的matrix ISA、dynamicLDS及VGPR已核对，是一条mapping线索。现有资料不能判断单一主瓶颈、bank conflict或精确occupancy；不因此推荐降低精度。
