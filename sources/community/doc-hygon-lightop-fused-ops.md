---
id: doc-hygon-lightop-fused-ops
title: 海光 lightop：融合算子库（RMSNorm/量化/attention 融合）的源码契约
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- rmsnorm
- normalization
- precision
- kernel-fusion
confidence: source-reported
date: '2026-10-29'
captured_at: '2026-10-29'
url: https://github.com/HYGON-AI/lightop/tree/main/python/lightop
source_commit: ffee6c2
---

## 审阅范围

本页记录对 `lightop` 仓库的直接源码阅读，锁定 commit `ffee6c2`
（2026-10-28，main）。Apache-2.0，海光自研（非 DeepSeek fork）。审阅等级：
direct-source-read（RMSNorm 家族 kernel + Python 绑定 + vllm-plugin-das
消费点抽查）。本页**没有**本机编译、数值对照或计时。

仓库规模：`csrc/` 121 个条目（norm/quant/moe/marlin/attention/loss 等），
`python/lightop/` 约 90 个模块。本页聚焦 GLM-5.3-Flash 推理路径实际用到的
RMSNorm/量化/attention 融合家族；GEMM/MoE/采样等家族只列目录不展开。

## RMSNorm 家族（csrc/rmsnorm/，22 个文件约 18600 行）

基础 kernel `rmsnorm.cu` 的 `rmsnorm_forward_kernel_fast<scalar_t, VEC,
blocksize>`：

- 每行一个 CTA，`memory::aligned_vector` 向量化（VEC 1/4/8/16），
  BlockReduceSum 归约 + `s_rstd` 共享广播，两遍读 x。
- **dispatch 按列数分档**（`rmsnorm_forward`，行 132-160）：
  cols≤128→`<.,1,128>`；≤1024→`<.,8,128>`；≤2048→`<.,8,256>`；
  ≤4096 且 rows>1200→`<.,8,512>` 否则 `<.,4,1024>`；≤8192→`<.,8,1024>`；
  ≤16384→`<.,16,1024>`。cols 必须 ≥64 且 %16==0。
- 训练模式可选输出 `rstd`（fp32）。

该家族的其余融合变体（目录名即功能）：`fuse_rms_add_opt`（残差 + 限幅
  `limit` 参数）、`fuse_rmsnorm_rope_qkv`、`fuse_rmsnorm_rope_quant_qkv`、
  `rms_norm_dynamic_per_token_quant`、`rms_norm_fp8_quant`、
  `rms_norm_smooth_per_token_quant`（554 行）、`fuse_gammarmsnorm_rope_qkv`
  等，覆盖 norm→rope→quant→KV store 的各种组合。

## 与 GLM-5.3-Flash 推理路径的绑定（vllm-plugin-das 消费点）

- **FusedRMSNormAndQuant**（`vllm_hcu/ops/fuse_rms_norm_quant.py:65`）：
  `lightop.norm.rms_norm_dynamic_per_token_quant(input, weight, epsilon,
  quant_dtype, residual, update_input)` —— RMSNorm + 逐 token 动态 INT8
  量化 + 原地残差更新，一个 kernel 完成（`python/lightop/rmsnorm.py:211`）。
- **SGLang 兼容变体**（`rmsnorm.py:234`）：`rms_norm_dynamic_per_token_quant_sglang`
  单独实现，docstring 声明与历史 kernel 的数值差异：量化观察的是
  **BF16/FP16 舍入后的** RMSNorm 输出、scale 用 `max(absmax,1e-10)/127`、
  fp32 残差和无中间 bf16 cast、支持 smooth_scale/clamp/residual_output。
  这是"同一算子两套数值语义并存"的实例。
- **FusedQkvSplitRmsNormRopeAttention**（`attention_runtime.py:178`）：
  `lightop.attention.split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant`
  —— QKV 切分 + q/k RMSNorm + RoPE + KV store 量化一步完成；注释说明
  fused writer 曾用已废弃的 split-K/V cache ABI，现保留计算融合、写入走
  原生 HIPC op。
- **INT8 量化**（`int8_runtime.py:113,150`）：
  `lightop.quant.per_token_quant_int8` 与 `lightop.gemm_ops.hipblaslt_w8a8_gemm`。
- mHC 相关（`boltops.mhc`，非本仓库——见下"边界澄清"）。

## hc_mix_stats / hc_mix_sinkhorn（mHC 家族在海光的另一实现）

`csrc/hc_mix_stats.cu` + `hc_mix_sinkhorn.cu` 提供
`hc_mix_stats`、`hc_mix_stats_sinkhorn_bf16x3`、
`hc_mix_stats_sinkhorn_deepgemm`（export.cpp:1826-1832）。要点：

- `mmac` intrinsic 按 arch 分派：gfx928/gfx936 用
  `__builtin_amdgcn_mmac_f32_16x16x16bf16`，**gfx938 用
  `__builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts`**（HCu 后缀指令族）。
- bf16 转换 gfx938 用 `__builtin_hcu_cvt_bf16_f32`（RNE），旧 gfx 走软件。
- 8 wave/CTA（4 个 K 子 slice × 2 个 N tile），slice 256。
- 在 vllm-plugin-das 上游 main 里**没有**找到这三个 op 的调用点——
  它们服务的可能是 SGLang 或其他部署路径（本仓库 README 不含 GLM 绑定说明）。
  与 mhc.py 的 TileLang/boltops 路径关系待运行时验证。

## deepgemm-das 对 lightop 的代码复用

deepgemm-das `csrc/include/exception.h` 内嵌了 `LIGHTOP_HOST_ASSERT` /
`LIGHTOPException` 定义（`#ifndef` 守卫），其 gfx938 kernels（含
tf32_hc_pernorm_gemm）的 host 断言全用它——两仓库共享同一套 host 工具宏
（源码复制，非包依赖）。

## 边界澄清：lightop ≠ boltops

vllm-plugin-das `mhc.py` 的 AITER_MHC 路径 import 的是 `boltops.mhc`
（`mhc_fused_tilelang`/`mhc_post_fwd`/`mhc_pre_big_fuse`/`pre_big_fuse_tilelang`）。
`boltops` 是部署环境里独立的 "AI Compiler ops (Triton & TileLang)" 包
（本机容器 `/data2/.../ds0251-site/boltops/`，含 fa/fla/gemm/ihc/mhc/mla 等
tilelang 子包），**不是 lightop 仓库的一部分**，也不在 52 个 HYGON-AI
社区仓库列表中。引用 mHC 的 boltops 实现时不要归到 lightop。

## 证据限制

- 本页为 source-level 阅读；**没有**本机 gfx938 上的编译、数值对照或计时。
- RMSNorm 家族 22 个 kernel 只精读了基础 `rmsnorm.cu`；融合变体的机制
  从文件名、Python 绑定签名与调用方注释确认，未逐行读 kernel 体。
- `hc_mix_stats*` 三个 op 在 vllm-plugin-das 上游 main 无调用点，实际服务
  的部署路径未确认。
- lightop 与 vllm-plugin-das 的版本配对（部署镜像里的 lightop 版本 vs
  上游 main `ffee6c2`）未核实；`rms_norm_dynamic_per_token_quant_sglang`
  这类数值语义敏感接口尤其需要以部署版本为准。
