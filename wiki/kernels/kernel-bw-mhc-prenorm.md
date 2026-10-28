---
id: kernel-bw-mhc-prenorm
title: mHC prenorm：GEMM 与行平方和的融合及分片
type: wiki-kernel
architectures:
- gfx938
tags:
- gemm
- normalization
- precision
confidence: source-reported
sources:
- doc-hygon-deepgemm-tf32-hc-prenorm
- doc-hygon-lightop-fused-ops
- doc-hygon-tilekernels-tilelang
date: '2026-10-30'
description: mHC pre block 的优化重点是让 mixes 投影 GEMM 与 RMSNorm 分母的行平方和共享同一次激活读取，并用 K 分片在小批量下填满 SM。
kernel_types:
- gemm
- normalization
languages:
- hip-cpp
- triton-rocm
techniques:
- kernel-fusion
- occupancy-tuning

related:
- pattern-fp32-staging
- kernel-bw-rmsnorm
---

mHC pre block 的优化重点是让 mixes 投影 GEMM 与 RMSNorm 分母的行平方和共享同一次激活读取，并用 K 分片在小批量下填满 SM。

## 计算组织

mHC pre block（hyperconnection）每个 token 做：`mixes = RMSNorm(x) @ fn.T`
（hc_mult3 × hc_hidden_size 的瘦 GEMM）、sigmoid pre/post 门、Sinkhorn 列归一化、
`layer_input = Σ_m pre[m]·residual[m]`。朴素实现会把 x 升 fp32 物化两次
（一次 GEMM 一次平方和）、中间 mixes 落盘再读回，host 侧还有十余个小 kernel。

## 优化思想

- **GEMM 与 sqr_sum 融合**：sqr_sum 是纯向量路径，不经过 MMA；把它放进
  GEMM kernel 的 A-tile 加载/转换循环，A 的 HBM/LDS 读就只发生一次。
  上游 deepgemm `tf32_hc_pernorm_gemm` 即此结构（TF32 16x16x8 MMA +
  在线 `sqr_sum_acc`，见来源页）。
- **split-K 以 token 数为开关**：瘦 GEMM 的 M=token 数在 decode/短 prefill
  时很小，M 向并行不足以填 SM；把 K（hc_hidden_size）切成 num_splits 份，
  partial fp32 直接落盘由后继 kernel 求和。上游用 SM 数与 k 块数归一
  split 到 (1,2,4,8)，token ≤ 16 时干脆退出 GEMM 路径改逐元素 FMA 融合。
- **后继融合**：rsqrt/sigmoid/Sinkhorn/加权求和收敛进一个 token 级
  kernel（上游 TileLang `mhc_pre_big_fuse`），GEMM 只留下投影 + 平方和。
- **精度分层**：mixes 投影可吃 TF32（尾数 10 bit），sqr_sum 走 fp32 向量
  路径不受 TF32 影响；两部分的精度特征不同，评估时要分开。

## 适用边界

- 融合 GEMM 的 N 侧很窄（hc_mult3=24 量级），tile 参数与 `n%8==0`、
  `k%64==0` 这类约束都为该形状族特化；N 变宽时 A 驻留寄存器/共享的策略
  需要重选。
- split-K 的 partial 求和发生在 fp32，语义安全；但 partial tensor 的额外
  显存与后继读回是固定开销，大 token 数时 split 应退回 1。
- TF32 只对投影合法的前提是上游训练/参考也容忍该精度；迁移前需确认
  模型侧的精度预算。

## 本机范围

本页机制来自上游源码阅读（source-reported），本机没有任何 mHC prenorm
相关 kernel 的编译、数值或计时记录。上游 kernel 自身标注了未完成项
（输出标量写回），说明该路径仍有优化空间，可作为本机选题时的候选假设，
但每个假设仍需在本机原 Task 上配对验证。

值得注意：TileKernels-das 提供了同一算子族的 TileLang 完整实现
（split-K stage0/stage1 + `use_tf32=True` GEMM + HCU 专属 smem
swizzle + 手写两级预取链，且带训练反向与 TF32 对齐的 torch 参考），
比 vllm-plugin 内联快照更新；做本机对照实验时应以它为社区基线
（见 doc-hygon-tilekernels-tilelang）。它与 deepgemm
tf32_hc_pernorm_gemm 的共同设计是 sqr_sum 不过 MMA、走纯向量路径
共享 A-tile 读取。
