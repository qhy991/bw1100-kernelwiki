---
id: exp-fp32-staging
title: 严格 FP32 MoE 矩阵指令和 dynamic LDS 的资源核对
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: experimental
sources: []
date: '2026-10-05'
description: 保留的MoEv33源绑定_cake_moe_gus_v33_t2048及down/t2080系列。
evidence_root: /data3/testuser01/experiments/bw1100-cake-compiler-completion-20261005
artifacts:
- FP32-STAGING-TRIAGE.md
evidence_scope: partial-runtime-triage
compiler: retained MoEv33 original bound source
limitations:
- resource_lead_not_bottleneck
- no_new_primitive_promotion
---

保留的MoEv33源绑定_cake_moe_gus_v33_t2048及down/t2080系列。
TTGIR/native/profile显示真正v_mmac_16x16x8_f32，不是scalar-FMA fallback。
GU metadata.shared32768、8组；TTGIR有8 local_alloc/8 local_load；native174 next-freeVGPR、30SGPR、0fixedLDS。
对应实际rowLDS32768/wgr512/wave64/VGPR176/SGPR32。

0 fixed LDS不等于无LDS：dynamic分配由dispatch证明。逻辑寄存器估计也不代替实际metadata。
当前是耦合register/shared资源线索，尚不能判断dominant LDS bottleneck、bank conflicts、exact occupancy或speedup。
先做同精度tile/width/pipeline mapping并保留paired全调用，才考虑Compiler缺口；无需凭slow添加dot primitive或降低FP32。
