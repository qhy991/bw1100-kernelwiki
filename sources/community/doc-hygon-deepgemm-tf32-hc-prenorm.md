---
id: doc-hygon-deepgemm-tf32-hc-prenorm
title: 海光 deepgemm-das tf32_hc_pernorm_gemm：mHC prenorm GEMM+sqr_sum 的源码契约
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- gemm
- precision
- runtime-dispatch
confidence: source-reported
date: '2026-10-29'
captured_at: '2026-10-29'
url: https://github.com/HYGON-AI/deepgemm-das/tree/main/csrc/arch/gfx938
source_commit: 3910d3a
---

## 审阅范围

本页记录对 `deepgemm-das` 仓库中 `tf32_hc_pernorm_gemm` 算子的直接源码阅读，
锁定 commit `3910d3a`（2026-10-26，main）。上游为 DeepSeek DeepGEMM
（基线 commit `559d79f`），海光分支在其上增加 INT8/perChannel 量化路径与
本算子。审阅等级：direct-source-read（kernel + host 入口 + Python 绑定 +
调用方 vllm-plugin-das 的 mhc.py，同日上游 main `abaac95`）。本页**没有**
本机编译、数值对照或计时。

阅读的文件：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `csrc/arch/gfx938/tf32_hc_pernorm_gemm.cu` | 426 | kernel + impl + host 入口 |
| `csrc/include/tf32_hc_pernorm_gemm.h` | 薄声明 | `tf32_hc_pernorm_gemm(A,B,D,sqr_sum,num_splits)` |
| `csrc/common/deepgemm_ops.cpp`（行 91-98） | — | 通用 TU 里的 `unsupported_unbuilt_op` stub |
| `deepgemm/tf32_hc_pernorm_gemm.py` | 15 | Python 薄绑定 |
| `op_test/test_tf32_hc_pernorm_gemm.py` | 91 | 上游自测（数值断言被注释） |
| `vllm-plugin-das/vllm_hcu/model_executor/layers/mhc.py`（行 11、588、1084） | 1406 | 生产调用方 |

## 算子语义与调用契约

kernel 计算两样东西（API 文档与测试一致）：

- `d ≈ a.float() @ b.T`（a 是 bf16 `[M,K]`，b 是 fp32 `[N,K]`，d 是 fp32）
- `sqr_sum ≈ a.float().square().sum(-1)`（逐行平方和，mHC RMSNorm 分母来源）

`num_splits` 给定时输出为分片布局 `[num_splits, M, N]` / `[num_splits, M]`，
由上层 `sum(0)` 合并（fp32 partial 相加，无单独 combine kernel）。
注意输出实际按扁平 `[num_splits*M, N]` 分配（测试 docstring 行 9）。

host 入口约束（`tf32_hc_pernorm_gemm.cu:399-424`）：

- a=BFloat16、b/D/sqr_sum=Float；D 行主序（`check_major_type_cd`）；
  sqr_sum 连续；`m==0` 直接返回。
- impl 级约束（行 359-368）：**`n <= 32 && n % 8 == 0`**、`k % 64 == 0`。
  `block_n = align(n,16)`，`n<=16` 与 `n>16` 两个模板实例
  （`<64,64,16|32,16,16|32,64,4,1>`，`__launch_bounds__(1024)`）。
- grid = `num_splits * ceil(m/64)`，block = 4 wave64（256 线程）；
  `kNumStages` 固定 4（受 `k/64` 上限裁剪）；动态 LDS 仅 `64*64*2 = 8 KiB`
  （只缓存 A tile；B 直接 buffer_load 到 VGPR，不过 LDS）。

## kernel 内部机制（源码直读）

- **MMA 路径**：`__builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts`（16x16x8 TF32
  matrix core），A operand 由 bf16 → fp32 标量转换后逐 `floatx2` 送入；每个
  K 步 8 次 `scalar2_array` 调用（行 185-234）。
- **A 加载**：`matrix_load_b16_lds_trans<64,16,1,0>` 直接全局→LDS 转置加载，
  读出用 `DS_READ_MATRIX_32X16_B16`（gfx938 专用，行 156-159）；越界 M 行用
  `nm_filter`（行 115-116）置零。
- **B 加载**：`buffer_load_reg_dwordx4`（带 `shape_k * actual_n` 越界钳制）。
- **sqr_sum 在线融合**：A tile 转 fp32 的同一个循环里累加
  `sqr_sum_acc_0 += x*x`（行 167、210）；每 warp 覆盖 `WARP_M=16` 行，
  最后 LDS 规约（4 lane 组求和，行 317-331）。**sqr_sum 不经过 MMA**，
  纯向量路径，因此与 GEMM 共享 A 的 HBM/LDS 读是本 kernel 的核心收益点。
- **流水线**：手写 vmcnt/lgkmcnt 计数（`vmcnt_wait((1+WARP_N/16*WARP_K/16)*(stages-1))`），
  前后各留 `stages-1` 步排空。
- **split-K 语义**：`kNumKBlocksPerSplit` 均分 K 块，余数块分给前面的 split
  （行 46-61）；每个 split 的 partial d/sqr_sum 写到 `split_m_offset = m*split_idx`
  偏移处。
- **已知不完善（源码注释自认）**：输出写回是标量路径
  （行 333 注释 “先用标量路径保证正确性”），没有用向量/矩阵 store——
  这是该 kernel 明示的后续优化空间。

## 调用方：vllm-plugin-das mhc.py 的 dispatch

`vllm_hcu/model_executor/layers/mhc.py`（上游 main `abaac95`）：

- `mhc_pre` 的 `is_rocm()` 分支（HCU 平台 `_enum = PlatformEnum.ROCM`，为真）
  走纯 torch 参考实现；**生产路径是 `mhc_fused_post_pre`**。
- `mhc_fused_post_pre`（行 908-1136）按 token 数分层：
  - `T <= 16`（decode）：`boltops mhc_fused_tilelang` FMA 路径
    （hc_mult3 个 dot 逐元素展开，不用 GEMM kernel）；
  - `16 < T <= 2048` 且 `hc_hidden_size ∈ (16384, 28672)` 且
    `VLLM_HCU_USE_TILELANG_MHC_PRENORM=1`（默认开）：TileLang split_k=32
    两段式 prenorm kernel；
  - **其余（含 `T > 2048`）：本算子 `tf32_hc_prenorm_gemm`**（行 1084），
    `n_splits = compute_num_split(64, hc_hidden_size, ceil(T/64))`，
    归一到 (1,2,4,8)，用 SM 数和 `k/64>=4` 下限约束（行 48-58）。
- 后接 `mhc_pre_big_fuse_tilelang`（或 boltops 版）：完成 RMSNorm rsqrt、
  sigmoid 门、Sinkhorn 归一和 `layer_input` 加权求和——即本算子只负责
  pre block 里"mixes 投影 + sqr_sum"的 GEMM 部分。
- `VLLM_HCU_USE_AITER_MHC`（默认开）只替换 post/big_fuse 的变体
  （boltops vs 本仓库 tilelang 版），**不影响 prenorm GEMM 选路**。
- 环境开关默认值见 `vllm_hcu/platforms/envs.py:46-48`。
- GLM-5.3-Flash 尺寸代入（hc_mult=4、hidden_size=4096）：hc_mult3=24、
  hc_hidden_size=16384，满足 `n=24 ≤ 32 且 24%8==0`、`k%64==0`——
  正是该 kernel `n≤32` 特化所服务的形状族。

## 与 lightop 的代码复用

deepgemm-das 的 `csrc/include/exception.h` 自带 `LIGHTOP_HOST_ASSERT` 宏与
`LIGHTOPException`（命名直接来自 lightop，行 25-33），`tf32_hc_pernorm_gemm.cu`
的 host 断言全部用它。即两个仓库共享同一套 host 侧工具宏（复制粘贴式复用，
非链接依赖）。

## 证据限制

- 本页为 source-level 阅读；**没有**本机 gfx938 上的编译、数值对照或计时。
- 仓库 `op_test/test_tf32_hc_pernorm_gemm.py` 的精度断言（`s < 1e-5`、
  `d < 1e-3`）被注释掉，只把 max_diff 写 CSV——上游测试**不强制**数值门槛。
- TF32 MMA 的精度（10 bit 尾数）意味着 `d` 与 fp32 参考的偏差由硬件决定；
  sqr_sum 是纯 fp32 向量路径，不受 TF32 影响。二者精度特征不同，迁移到
  精度敏感场景时要分开评估。
- `prefill_kernel_opt_readme` 式的性能记录本仓库没有对 tf32_hc 算子给出
  上游自报数字；README 性能声明仅覆盖 GEMM 家族接口。
- 深层上游（DeepGEMM 原仓库）没有本算子，无法做跨实现对照。
