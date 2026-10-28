---
id: doc-hygon-aiter-fla-hip-source
title: 海光 AITER FLA HIP 实现：chunk/decode gated delta rule 的源码契约
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- aiter
- fla
- attention
- precision
confidence: source-reported
date: '2026-10-28'
captured_at: '2026-10-28'
url: https://github.com/HYGON-AI/aiter-das/tree/main/csrc/fla
source_commit: 47d0248
---

## 审阅范围

本页记录对 `aiter-das` 仓库 `csrc/fla/` 目录的直接源码阅读，锁定 commit
`47d0248`（2026-10-20，main）。审阅等级：direct-source-read。阅读的文件：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `csrc/fla/chunk_gated_delta_rule_fwd.cu` | 804 | ATen 包装、运行时 dispatch、BV16/32/64/128 路由 |
| `csrc/fla/chunk_gated_delta_rule_fwd_kkt_solve.cu` | 223 | KKT solve 独立 host 入口 |
| `csrc/fla/fused_recurrent_gated_delta_rule_packed_decode.cu` | 673 | packed batch decode 递归 kernel |
| `csrc/fla/fused_sigmoid_gating_delta_rule.cu` | 638 | 融合 sigmoid 门控 decode 递归（GLM KDA 语义） |
| `csrc/fla/chunk_fwd_o.cu` | 334 | chunk 输出投影 |
| `csrc/fla/fla_api.cpp` | 197 | C++ API 聚合 |
| `csrc/fla/docs/bv32_bv64_kernel_design.md` | 1525 | BV32/BV64 kernel 设计文档 |
| `csrc/fla/instances/` | 20 个 TU | (input dtype, state dtype, BV) 模板实例 |

Python 入口在 `aiter/ops/fla.py`（行 124-344）：`vllm_fused_sigmoid_gating_delta_rule_update`
（行 290）、`aiter_fused_recurrent_gated_delta_rule_packed_decode`（行 330）、
`chunk_gated_delta_rule_fwd_kkt_solve_hip`（行 179）。JIT 模块名 `module_cpp_api`。

本页只覆盖源码事实与文档转述；**没有本机编译、数值对照或计时**。上游设计文档中的
性能数字按 source-reported 转述，不作为本机结论。

## chunk_gated_delta_rule_fwd 的接口契约

源码 `chunk_gated_delta_rule_fwd.cu`：

- **Shape 特化**：只支持 `headDimK == 128`、`headDimV == 128`、`chunk_size == 64`、
  `transpose_state_layout == True`（行 226-227、402-403、576-577、600-603 的 TORCH_CHECK）。
  `transpose_state_layout=false is not yet implemented`。
- **状态布局**：`initial_state` / `final_state` 为 `(state_rows, H, V, K)`——V 在前
  K 在后，且必须 H/V/K 维 dense（行 604-607 检查 stride(3)==1、stride(2)==K、
  stride(1)==V*K）。`h`（chunk 级隐状态快照）固定分配为 `(B, NT, H, V, K)`，
  行 17-19 注释说明这是 Triton kernel 硬编码布局。
- **vLLM/SGLang 语义差异**（行 21-30、642-654、741-750）：
  - vLLM 入口返回 `(h, v_new, final_state)`，`final_state` 是独立分配的 FP32 tensor；
    `initial_state` 必须是 FP32。
  - SGLang 入口把 `ht_ptr` 别名到 `initial_state.data_ptr()` **原地更新**，
    `output_final_state` 参数被显式忽略（`inplace_final_state=true` 硬编码）；
    SGLang 的 state 允许 BF16 或 FP32。`initial_state` 与 `initial_state_indices`
    必须成对提供。
  - `save_new_value=false` 时 `v_new` 返回 `{0}` 尺寸占位 tensor，保持返回元数稳定。
- **dtype**：k/w/u 必须同为 fp16 或 bf16 且最后一维连续；`g`（标量 per-head gate）
  与 `gk`（key-wise gate）**必须 FP32**（行 455-468）。vLLM persistent state 必须
  FP32；GQA 要求 `H % Hg == 0`。
- **varlen 契约**（行 502-541、656-665）：`cu_seqlens` int32/int64（IndexT 只由它
  决定）；`chunk_indices` 只用 `size(0)` 取 NT，**kernel 不读其元素**，dtype 可与
  cu_seqlens 不一致；`chunk_offsets` 强制 int64 且必须提供。建议上层预计算缓存。
- **索引越界责任在调用方**：`initial_state_indices` 的范围检查被 `#if 0` 禁用
  （行 618-628，注释说明为避免 `.item()` 每次调用同步），语义上跟随 Triton。

## BV 自动选择与 gfx938 阈值

`fla_select_bv_gfx938_72`（行 61-67）针对 72-CU gfx938 的经验阈值，`P = N*H`
（N 为 padded batch 或 varlen 序列数，H 为 value head 数）：

| P | BV | CTA 数 |
|---|---|---|
| ≤ 9 | 16 | 8P |
| 10–18 | 32 | 4P |
| 19–36 | 64 | 2P |
| ≥ 37 | 128 | P |

- `AITER_FLA_FORCE_BV=auto|0|16|32|64|128` 可强制固定做 A/B（行 239-266）；
  强制 64/128 要求 gfx938 + transpose_state_layout + K=V=128。
- 非 gfx938 设备保留旧规则：gfx936 P<12→BV16 否则 BV32；120-CU gfx92a 实测
  边界 P<16（设计文档 README 节，注明依据 120-CU K200_AI 上的成对 GPU-graph
  benchmark，并声明是单阈值启发式，长序列大 P 可能重新偏好 BV16）。
- 捕获的 CUDA graph 保留捕获时选定的 BV。

## kernel 内部机制（设计文档转述，source-reported）

`docs/bv32_bv64_kernel_design.md` §18 是当前生产 BV64 LDS32 设计：

- 递推数学（§2）：GEMM0 `P = W_c @ H_low^T`，residual `R_raw = U - P`，
  row gate `exp(g[last]-g[t])`，GEMM1 `Delta = R_gate^T @ K_c`，
  state 更新 `H = H*state_scale + Delta`，同时输出更新前的低精度 state 快照。
- 精度边界（§18.4）：state 在 chunk 间常驻 FP32 VGPR，仅 GEMM0 operand 与
  h/final store 转 BF16；`v_new` 保存 gate **前**的 raw residual；SafeNatural
  指数只用于 row scale，旧状态 `exp(g[last])`/`exp(gk[last,k])` scale 走非 safe 路径。
- 资源（§18.5，gfx938 code object）：VGPR 128、SGPR 81、0 spill、动态 LDS
  32768 B/CTA、256 线程、occupancy 2 CTA/CU。LDS 32 KiB 三段 alias：
  [0,8K) W→U→padded V、[8,16K) W→V stages、[16,32K) K tile。每线程持续保留
  32 个 FP32 state 值；`group_segment_fixed_size=0` 是因为 LDS 动态传入。
- 流水线（§18.3）：13 步 steady chunk 顺序，vmcnt(4)/vmcnt(12) 与
  lgkmcnt(0)+barrier 的计数依据在文档中有逐条论证。
- 上游自报数字（§18.6，非本机）：目标 shape 32 条长 1216 序列（T=38912、H=32、
  Hg=16、NT=608），warmup 25/rep 100，BV32 2.6726 ms → BV64 LDS32 2.0802 ms，
  speedup 1.285x、延迟降 22.2%；Triton 对照 `h/v_new/final_state` 通过
  `rtol=atol=0.08`。§12-17 是被取代的 28 KiB two-V32-pass 设计，保留作演进记录。

## decode：fused_sigmoid_gating_delta_rule_update

源码 `fused_sigmoid_gating_delta_rule.cu`，Python 入口
`aiter/ops/fla.py:290`。这是 GLM5Next KDA decode/spec-verify 语义的融合递归 kernel：

- **KDA 门控公式**（kernel 行 276）：`g = expf(-A * softplus_th(a + dt_bias))`，
  其中 `A = expf(A_log[hv])`（per-head，行 191），`a`/`dt_bias` 为 per-channel。
  `softplus_th`（行 104-107）带 threshold 分段：
  `beta*x <= th ? log1p(exp(beta*x))*inv_beta : x`，默认 threshold=20。
- 非 KDA 路径用标量 gate：`g = expf(-A * softplus_th(a_scalar + dt_scalar))`。
- **beta 门**：`beta_gate = 1/(1+expf(-b))`（sigmoid，行 253 附近）。
- **内嵌 q/k l2norm**：模板参数 `UseNorm`，`rsqrtf(sum(x^2)+1e-6)`，归约在
  GroupSize 内用 `__shfl_xor` 完成（行 285-296）。
- 归约与输出：`coef = (v - hk*inv_k) * beta_gate * inv_k`，输出
  `out * inv_q * scale`（行 316-345）。
- **spec-verify 支持**：`num_accepted_tokens`（决定从哪个 token 起步）与
  `idx_read`/`idx_write`（state scatter/gather 槽位索引），负索引跳过
  （行 153-176、365-385）。
- **stride-aware**：q/k 直接按 `p.H` stride 读，不要求 dense（对照上游
  vLLM 的 `.contiguous()` 物化路径）。
- state 始终 FP32 读写（f32x4 向量化）。

## 与 vllm-plugin-das 的对接

`vllm-plugin-das`（main，2026-10-28）的 `vllm_hcu/patch/worker/op_opt/`
里有直接绑定：

- `_aiter_fla.py:71` 把 `fused_sigmoid_gating_delta_rule_update` 映射到
  AITER 的 `vllm_fused_sigmoid_gating_delta_rule_update`。
- `patch_gdn_linear_attention.py`（行 44、202、264、365）在 GDN/线性注意力
  路径上替换 sigmoid 更新入口。
- `patch_glm5next_kda_strided_recurrent.py` 的 docstring 引用本 kernel 作为
  stride 设计参照（见 [doc-hygon-vllm-plugin-glm53-patches](doc-hygon-vllm-plugin-glm53-patches.md)）。

## 证据限制

- 本页是 source-level 阅读；**没有**本机 gfx938 上的编译、数值对照或计时。
- 设计文档中的 VGPR/LDS/occupancy 与 1.285x 数字是上游自报（source-reported），
  未在本机复测；其目标 shape（H=32）与 GLM5Next（H=64/t）不同，不能直接迁移。
- `aiter/ops/triton/fla/` 下的 Triton 参考实现（vLLM/SGLang chunk_delta_h.py）
  未逐行阅读。
- 上游 main 与本机容器镜像内置版本可能不同（vllm-plugin 侧已核实存在差异）；
  使用前应以实际部署版本的 SHA 为准。
