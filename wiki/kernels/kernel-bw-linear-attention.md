---
id: kernel-bw-linear-attention
title: Chunk gated delta rule：混合 gate 与 FLA component
type: wiki-kernel
architectures:
- gfx938
tags:
- attention
- precision
- fla
- lds
confidence: experimental
sources:
- exp-community-baselines
- exp-night-exclusions
- doc-hygon-aiter-fla-hip-source
- doc-hygon-vllm-plugin-glm53-patches
- doc-hygon-flashmla-sparse-mla
date: '2026-10-29'
description: Chunk gated delta rule 的优化需要先对齐每个阶段的gate语义，再选择state更新的块大小。
kernel_types:
- attention
- reduction
languages:
- triton-rocm
- python
- hip-cpp
related:
- pattern-precision-not-output-only
- pattern-storage-rebinding
---

Chunk gated delta rule 的优化需要先对齐每个阶段的gate语义，再选择state更新的块大小。

## 计算组织
chunk结构把序列划分为局部计算和跨chunk状态更新。不同site可能使用不同gate；head映射、padding和状态精度都会影响结果。单个同名fused入口不一定表达这个组合。

## 优化思想
复用已优化的组件kernel，按原数学连接各阶段。state的V tile、执行组和pipeline深度共同决定寄存器/LDS footprint；较小tile可减少单CTA资源，但会增加tile数或重复工作。
图重放可以减少host dispatch，若每次把当前输入复制到owned buffer，copy成本也应计入完整调用。

## HIP 参考实现对照（上游源码，非本机实测）
海光 AITER 公开了这套算子的 HIP 实现（csrc/fla，见来源页 doc-hygon-aiter-fla-hip-source）。
与本页机制直接相关的上游事实：

- chunk 前向只特化 K=V=128、chunk 64、`transpose_state_layout=True`；状态是
  (N, H, V, K) 且 H/V/K 维必须 dense。这是确认 chunk 组件状态映射时的权威布局，
  而不是本机 FLA Triton 链的隐式假设。
- 递推拆成两次 GEMM（projection `W @ H_low^T` 与 delta `R_gate^T @ K`）；
  state 在 chunk 间常驻 FP32 寄存器，仅 GEMM operand 与快照 store 转 BF16，
  `v_new` 保存 gate 前的 raw residual——舍入位置与"只在输出处看精度"不同。
- BV（V tile 宽）按 P=N*H 分档选 CTA 数（72-CU gfx938：P≤9→16、≤18→32、
  ≤36→64、其余 128），本质是在"每 (sequence, head) 的公共 W/K/G 重复加载次数"
  与单 CTA 资源之间做交换；`AITER_FLA_FORCE_BV` 用于 A/B。上游在
  LDS 32 KiB 三段 alias + 每线程 32 个 FP32 state 的设计上自报 1.285x
  （BV32→BV64，H=32 目标 shape，source-reported，本机未复测，不可直接套到 H=64）。
- decode/spec-verify 的 GLM KDA 门控有融合 HIP kernel：per-head A_log、
  per-channel dt_bias、`exp(-A·softplus(a+dt_bias))`、sigmoid beta、q/k l2norm
  与 state scatter/gather 全在一个 kernel 内；q/k/v 按 stride 直接读，省去
  上游 `.contiguous()` 的 4-5 个 copy（见 doc-hygon-vllm-plugin-glm53-patches）。
  该收益是源码声明的机制，尚无本机或上游测量数字。

GLM-5.3-Flash 里与 KDA 线性注意力层交替的 DSA 层走 sparse MLA 路径，海光
社区另有整套 FlashMLA-das 实现（token 级稀疏 prefill/decode、attn_sink 尾部
融合、topk_length 变长处理、索引 LDS 预取），其源码契约见
doc-hygon-flashmla-sparse-mla；两者同属"混合注意力模型的注意力层"对照面。

对照含义：本机 FLA Triton 组件链的下一步有界改写，可以按上游已验证的
方向（BV 分档、公共输入跨 V tile 复用、stride 直读）提出假设，但每个假设
仍需在本机原 Task 上配对验证；上游数字只用于选题，不作为本机预期加速。

## 本机范围
已有未改动FLA组件的FP32 mixed-gate组合资格，以及限定state映射的profile。它纠正的是早期adapter判断，不表示任意fused FLA入口兼容。未独立确认的搜索分数只保留为候选记录。
上游 HIP 实现的对照是源码级证据；本机没有任何 AITER FLA HIP kernel 的编译、
数值或计时记录。
