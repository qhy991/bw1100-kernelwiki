---
id: doc-hygon-tilekernels-tilelang
title: 海光 TileKernels-das：TileLang 算子库（mHC/Sinkhorn/MoE/量化）的源码契约
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- normalization
- moe
- precision
- runtime-dispatch
confidence: source-reported
date: '2026-10-30'
captured_at: '2026-10-30'
url: https://github.com/HYGON-AI/TileKernels-das/tree/main/tile_kernels
source_commit: 7ee2019
---

## 审阅范围

本页记录对 `TileKernels-das` 仓库的直接源码阅读，锁定 commit
`7ee2019`（2026-10-23，main，近头）。上游为 DeepSeek
`deepseek-ai/TileKernels`（基线 commit `36d9e45`，主分支），海光
适配为 tilelang-hygon 后端 + 架构细节修正。构建依赖
TileLang ≥1.12.0、PyTorch ≥2.10，声明支持 BW1000/BW1100/BW150/
K100_AI（DTK 栈）。审阅等级：direct-source-read（mHC 家族全量 +
quant/engram/moe 抽样 + config + 测试布局）。本页**没有**本机
编译或运行。

阅读的文件：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `README.md` | — | 范围声明、上游基线 |
| `tile_kernels/mhc/pre_big_fuse_kernel.py` | 165 | mHC pre 大融合 kernel（norm+mix+Sinkhorn+apply） |
| `tile_kernels/mhc/pre_norm_fn_splitk_kernel.py` | 149 | pre GEMM+sqrsum split-K 双 stage |
| `tile_kernels/mhc/norm_fn_kernel.py` | 307 | norm_fn 前向/反向、`use_tf32` GEMM |
| `tile_kernels/modeling/mhc/ops/pre_big_fuse.py` | 256 | 调度启发式 + TF32 舍入 + 组装 |
| `tile_kernels/mhc/post_kernel.py` | 342 | mHC post 前向（comb×b + c·d） |
| `tile_kernels/mhc/multilayer_recompute_kernel.py` | 197 | 多层重算（指针表 + `-ffp-contract=off`） |
| `tile_kernels/config.py` | 45 | SM 数与每 SM 共享内存探测 |
| `tile_kernels/torch/mhc.py` | — | torch 参考实现（含 TF32 软件仿真） |
| `tile_kernels/quant/`、`engram/`、`moe/` | — | 目录级清点 + 抽样 |
| `tests/mhc/` | — | 12 个测试文件布局 |

## mHC 家族（本页核心）

### pre_big_fuse：一个 kernel 吞掉整个 mHC pre

`_mhc_pre_big_fuse`（TileLang `@T.prim_func`，threads=128，
每 token 一个 block）在**单个 kernel 内**串起：

1. `_mhc_pre_norm_fn_fwd_norm`——对 GEMM 部分
   （`gemm_out_mul`，split-K 部分和）做 warp 间归约 + rsqrt 得
   rms，再乘 mixes；
2. `_mhc_pre_split_mixes_fwd`——post/comb/pre 三组 mix 的
   sigmoid×scale+base；
3. `_mhc_sinkhorn_fwd`——4×4（mhc_mult=4）矩阵的行/列归一化
   Sinkhorn 迭代（`sinkhorn_repeat` 次行-列交替，softmax 起步）；
4. `_mhc_pre_apply_mix_fwd`——`layer_input = Σ_i pre_mix[i] ×
   residual[:, i, :]`。

**warp 分工**（同一 block 内）：`tx < 64` 的两个 warp 算
post_mix/comb_mix/Sinkhorn；`tx ≥ 64` 的 warp 做 pre_mix 与
residual 累加（hidden 按 `gcd(512, hidden)` 分块，
`num_stages=0` 串行）。mixes 经 `T.alloc_shared` 广播给两组。
split-K 部分和的归约按 `n_splits` 分三档（≥4 且整除分组归约 /
≥2 串行累加 / =1 直取），对应下游 GEMM 的三种发射形态。

### pre GEMM：TF32 + split-K + sqrsum 融合

`mhc_pre_gemm_sqrsum_splitk_kernel`（stage0/stage1 两个 kernel）：

- **stage0**：grid `(split_k, ceildiv(tokens, token_block))`，
  bf16 的 x 与 fp32 的 fn 做 `T.gemm(..., transpose_B=True,
  k_pack=2, use_tf32=True)`，同时在内积循环旁以纯向量 FMA 累加
  `x·x` 的 sqrsum（sqrsum **不过 MMAC**——与 deepgemm
  tf32_hc_pernorm_gemm 的设计一致，见
  [[doc-hygon-deepgemm-tf32-hc-prenorm]]）。共享内存放
  `(token_block, hidden_block)` bf16 + `(32, hidden_block)` fp32，
  用 `tilelang.layout.make_hcu_swizzled_layout(..., major_pack=2)`
  标注 HCU 专属 swizzle；x/fn smem 时分复用 32KB。手写预取链
  （pre→smem→frag 三级，`x_frag_pre` 提前搬运下一块）。
- **stage1**：64 线程 warp 归约 split-K 部分和（sqrsum 用
  `warp_reduce_sum`，out 逐 split 串行累加）。

### 调度启发式（ops/pre_big_fuse.py）

`get_block_info`（lru_cache，token 数按 32 对齐后做 key）按
`token_loop` 与 `cu_count` 比值分档选 `n_splits_pre`
（1/4/8/16/32/64）与 `token_block`（32/64/128），
`hidden_block` 128（occupancy 2；不整除时降 64）。
`use_small_token_splitk` 要求 `hidden_loop % n_splits == 0`，
否则整体退回单 split 路径（`_mhc_pre_norm_fn_fwd_mul`，同一
GEMM+sqrsum 融合但不切 K）。`cu_count` 来自
`torch.cuda.get_device_properties("cuda").multi_processor_count`
——注意在 HCU 环境下走 `device_type="cuda"` 的 vllm-plugin 平台
映射（[[doc-hygon-vllm-plugin-glm53-patches]]）。

### TF32 数值契约（commit 7ee2019 的主题）

`round_to_tf32`（TileLang kernel 与 torch 参考两版）做
**FP32→TF32 网格舍入**：`(bits + 0x1000)`，TileLang 版仅加不掩码
（依赖后续截断），torch 参考版 `(bits + 0x1000) & -0x2000` 显式
清低位。参考实现 `_MHCPreNormFnRefTLAlign` 对 fn 做软件 TF32 量化、
GEMM 时**临时关掉** `torch.backends.cuda.matmul.allow_tf32`（用
FP32 累加保证参考确定性），保证与 TileLang `use_tf32=True` kernel
的舍入点对齐。相关修正历史：`525c3df` 关 fp contraction（重算
kernel `-ffp-contract=off`）、`7ee2019` 补齐参考的 TF32 仿真、
`7d8b1fd` 关前向 kernel 的 SLP 向量化。

### 与 vllm-plugin-das 的关系（源码级对照）

vllm-plugin-das `mhc.py` 里 `mhc_pre_big_fuse_tilelang` /
split_k=32 TileLang 路径与本仓库**同源但独立内联**：
vllm-plugin 把简化版 kernel 直接写进
`vllm_hcu/model_executor/layers/mhc.py`（无 `round_to_tf32` 预处理、
`make_swizzled_layout` 而非 `make_hcu_swizzled_layout`、
`n_splits` 由 SM 数归一化为 1/2/4/8 而非 64 档表、`use_tf32` 未开）；
本仓库版本更新（HCU 专属 swizzle、TF32 契约、两级预取链）。
两个仓库互不 import（grep 无 tile_kernels 引用）——部署上
vllm-plugin 用的是 boltops 包内的另一份拷贝（见
[[doc-hygon-lightop-fused-ops]] 的 boltops 边界澄清）。即 mHC
TileLang 实现存在 **TileKernels-das（上游演进）与
vllm-plugin 内联（部署快照）两条线**。

## 其他家族（目录级）

- **quant/**（16 文件）：per-token / per-channel / per-block 的
  FP8/FP4/E5M6 cast，含 swiglu 前后向融合量化、无损 per-block
  cast。`float4_e2m1fn` 走 CUDA 兼容路径（tilelang 抽象屏蔽）。
- **moe/**（14 文件）：top-k 门控、group_count、fused expand/
  reduce、权重归一化、TP mask。`get_fused_mapping_kernel.py`
  有显式 `is_hcu = torch.version.hip is not None` 的 warp=64 vs
  32 分支（注释 "HCU/HIP wavefront is 64, CUDA warp is 32"）。
- **engram/**（6 文件）：Engram 门控（fused RMSNorm 前后向、
  weight 梯度归约、hash）；`7d8b1fd` 修了 SLP 向量化、`ad68d05`
  修 async copy group 的 wait 计数——engram 是 commit 修复活动
  最密集的家族。
- **transpose/**：batched transpose。
- **modeling/**：`torch.autograd.Function` 包装（engram 门控、
  mHC pipeline），训练可用。
- **config.py**：`get_max_smem_per_sm()` 显式绕开
  `shared_memory_per_multiprocessor` 属性（注释 "this prop
  seems not correct in hcu env"），按 gcnArchName 硬编码
  gfx936/938/92a → 64KiB、gfx946 → 128KiB。这是 HCU 环境下
  torch 属性不可靠的一个独立佐证。

## 测试布局

`tests/` 按家族分目录（mhc 12 文件、moe 9、engram 5、quant、
transpose），pytest + 自带 `pytest_benchmark_plugin.py`（`--run-benchmark`）
与 `--count` 压测插件；`TK_FULL_TEST=1` 全量压力测试。mHC 测试
对每个子算子有独立 torch 参考对照（`tile_kernels/torch/`），参考
实现带 TF32 仿真对齐。

## 对本机优化工作的含义

- 本仓库是 **mHC 完整算子族（含训练/反向）的 TileLang 权威实现**，
  比 vllm-plugin 内联版新；若 GLM-5.3-Flash 的 mHC prenorm 路径
  要做优化对照，应以本仓库的 split-K + TF32 + swizzle 版本为
  社区基线，而不是 vllm-plugin 的快照。
- `make_hcu_swizzled_layout(major_pack=2)`、`use_tf32=True` +
  手写预取链是 TileLang 在 HCU 上拿性能的三个已验证抓手
  （source-reported）；KDA 若走 TileLang 路线可直接复用这些
  pass_config 组合（`TL_DISABLE_WARP_SPECIALIZED`、
  `TL_PTXAS_REGISTER_USAGE_LEVEL=10`、`TL_DISABLE_VECTORIZE_256`、
  `TL_ENABLE_FAST_MATH` 的搭配在 mhc/engram 各 kernel 中反复出现）。
- `get_max_smem_per_sm` 的硬编码表是 HCU 环境探测的实用片段。

## 证据限制

- 本页为 source-level 阅读；**没有**本机编译或运行。README 所称
  "approach the limit of hardware performance" 与各处 occupancy
  注释均为 source-reported。
- quant/engram/moe 家族只做了目录清点与单文件抽样
  （per_token_cast、engram_gate、get_fused_mapping），未逐文件
  精读。
- TileLang 后端（tilelang-hygon）本身不在审阅范围内；
  `make_hcu_swizzled_layout` 的 swizzle 语义未追进 tilelang 源码。
- 与 vllm-plugin 内联版的差异对照基于两仓库锁定 commit 的静态
  阅读；两者是否会在上游合流无法从源码判断。
