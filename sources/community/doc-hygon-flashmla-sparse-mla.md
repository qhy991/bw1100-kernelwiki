---
id: doc-hygon-flashmla-sparse-mla
title: 海光 FlashMLA-das：sparse MLA prefill/decode（DSA 路径）的源码契约
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- attention
- flash-attention
- precision
- runtime-dispatch
confidence: source-reported
date: '2026-10-29'
captured_at: '2026-10-29'
url: https://github.com/HYGON-AI/FlashMLA-das/tree/main/csrc
source_commit: 1cea260
---

## 审阅范围

本页记录对 `FlashMLA-das` 仓库 sparse MLA 路径的直接源码阅读，锁定 commit
`1cea260`（2026-10-22，main）。上游为 DeepSeek FlashMLA（基线 commit
`48c6dc42`）。审阅等级：direct-source-read（API 层 + phase1 prefill kernel +
DSA MLS dispatch/adapter + BF16 DSA decode 入口 + Python interface 抽读）。
本页**没有**本机编译、数值对照或计时。

阅读的文件：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `csrc/api/sparse_fwd.h` | 280 | prefill API + FwdFeatures 特性开关 |
| `csrc/gfx93/prefill/sparse/phase1.cuh` | 1517 | 主 prefill kernel（B_H=64 模板） |
| `csrc/gfx93/prefill/sparse/config.h` | 158 | 两个 KernelTemplate 定义 |
| `csrc/gfx93/prefill/sparse/dsa_mls/dispatch.h` | 89 | DSA MLS gfx938 路由 |
| `csrc/gfx93/prefill/sparse/dsa_mls/params_adapter.h` | 125 | can_run/should_run/make_kernel_params |
| `csrc/gfx93/decode/sparse_bf16_dsa/fwd.cu` | 680 | BF16 DSA decode host 入口 |
| `flash_mla/flash_mla_interface.py` | 766 | Python 接口 + 索引排序预处理 |
| `prefill_kernel_opt_readme`（根目录） | 纯文本 | 2026-05-22 phase1 优化记录 |

## 仓库结构总览

- `csrc/gfx93/`：gfx936/gfx938 共用实现。`prefill/sparse/`（phase1.cuh 通用
  模板 + `dsa_mls/` 专用路径）、`decode/`（`sparse_fp8/` split-KV persistent、
  `sparse_bf16_dsa/`、dense 变体）。
- `csrc/gfx9/decode/`：combine kernel（splitkv 部分和合并）与
  `get_decoding_sched_meta`。
- `csrc/extension/`：gfx936 旧式 flash_fwd（bf16/fp16/fp8）与 gfx938 fp8
  decode（含预编译 `.co` 汇编对象 `flash_mla/asm/`）。
- 依赖 HYTLASS（pinned `4fad709b`，submodule）；hute/hytlass 的
  `GFX928_16x16x32_F32BF16BF16F32_NN` MMA atom 经 `hytlass_ext.h` 注册。

## Python 接口契约（flash_mla_interface.py）

`flash_mla_with_kvcache`（行 294）decode 入口：

- 稀疏模式要求 `causal=False`；`block_table`/`cache_seqlens` 可为 None
  （`indices` 内含页块号 + 页内偏移的绝对位置）。
- `topk_length`/`extra_topk_length`：只处理每序列最左 `topk_length` 个索引
  （相对 masking 省计算）；docstring 明示 NaN 边界情形（topk_length 之外的
  有效索引若指向含 NaN 的 K，输出会含 NaN）。
- `attn_sink`（`[h_q]` fp32）：输出乘 `exp(lse)/(exp(lse)+exp(attn_sink))`，
  不影响 lse/max_logits；`+inf` 使输出为 0。
- `extra_k_cache`/`extra_indices_in_kvcache`：附加 KV 池（PCP 公共前缀之外
  的补充 token），格式同主 cache。
- `num_splits` 仅 BF16 sparse decode 可覆盖，其余路径用 sched_meta。

`flash_mla_sparse_fwd`（行 436）prefill 入口：`q [s_q,h_q,d_qk]`、
`kv [s_kv,1,d_qk]`、`indices [s_q,1,topk]` int32；无效索引 -1 或 ≥s_kv；
返回 `(out, max_logits, lse)`，softmax 以 2 为底（`sm_scale*log2(e)`）。
**无 batch 维**，多 batch 由上层展平模拟。

**索引排序预处理**（行 101-205）：gfx936 且 d576、`h_q∈{64,≤16}`、topk=2048、
无 topk_length/attn_sink 时，对 indices 做 Triton 升序排序
（`tl.sort`，无效值→INT32_MAX→-1），提升 buffer_load 索引局部性；
`s_q≤128` 时跳过。另有把 topk_length 补满到 topk 的替代路径。

## phase1 prefill kernel（通用模板，phase1.cuh）

`KernelTemplate_B_H_64<D_QK, HAVE_TOPK_LENGTH, IS_TOPK_2048, USE_ATTN_SINK,
CACHE_INDICES_IN_LDS, SPLIT_KV, DECODE_C_LOAD>`：

- **tile**：B_H=64（一个 CTA 处理 64 个 q head 的 1 个 token 块）、
  B_TOPK=64、4 wave64（256 线程）；grid `(h_q/64, s_q/64, ...)`。
- **A（q）驻留 VGPR**：`q_reg[18]`（D_QK/32 个 `__bf16 x8`），一次
  `__builtin_amdgcn_buffer_load_dwordx4` 循环载入后全程复用。
- **索引取数**：K/V 行号来自 `gIndices[row]`；`IS_TOPK_2048 ||
  CACHE_INDICES_IN_LDS` 时先把 1024 个索引 `buffer_load ... , lds` 预取到
  LDS `sIndices`（行 154-187），topk=2048 分两个 1024 窗口换入。
- **手写 buffer_load→LDS**：K/V 都用 `s_mov_b32 m0` + `buffer_load_dwordx4
  ... idxen offen ..., lds` 直接全局→共享（带 stride 编码的 resource
  descriptor）；gfx938 用与 gfx936 不同的 lane→(row,col) 映射（`row_/col_`
  XOR swizzle，行 85-111）。`DECODE_C_LOAD` 变体改用普通 C 指针 + uint4
  拷贝并做行有效性检查（大 cache >32bit 寻址时）。
- **QK/PV 流水线**：`LOAD_K_AND_QK_GEMM`/`LOAD_V_AND_PV_GEMM` 宏展开的
  手工软件流水（vmcnt(3→0) 递减等待 + barrier）；QK 16 chunk、PV 16 n-tile。
- **softmax**：`softmax_rescale_o_prefill_4x1` 在线 rescale；输出 bf16 用
  `__builtin_hcu_cvt_pk_bf16_f32`（gfx938）打包。
- **attn_sink 尾部融合**（行 729-748）：稳定 sigmoid——
  `sink_delta=(lse-sink)*log2e`，`scale=(delta>=0?1:exp2(-|delta|))/(1+exp2(-|delta|))`
  再乘到 O 上；`+inf` sink 直接置 0。**在 kernel 内完成，不另发 kernel**。
- **topk_length 有效性**：无效 token 的 score 置 -INFINITY（行 459-465），
  行号钳制到 `s_kv`（哨兵行，行 136）。
- `prefill_kernel_opt_readme` 记录 2026-05-22 的 D_QK=512/H64/topk=512
  优化：QK 流水扩展到 16 chunk（原 576 专用 18-chunk 改造）、非 topk-2048
  路径避免索引 LDS 预取、attn_sink 缩放加入 D512 路径；上游自报（gfx936，
  s_kv≥8192 时）延迟降 47-64%，958 例回归全过。**dispatch 特意收窄到目标
  慢路径**以避免回归（readme Notes 节）。

## DSA MLS 路径（gfx938 专用，prefill 的替代实现）

`gfx93::fwd::dsa_mls` 是从 legacy MLA kernel（`legacy/src/flash_fwd_b16_mla.h`，
1151 行）包了一层 include 边界的实现。选路（`fwd.cu:29-32`）：
`enable_dsa_mls_prefill = is_current_device_gfx938()`，且 `should_run`：

| d_qk | h_q | topk | s_kv 阈值 |
|---|---|---|---|
| 512 | 64 | 512 | ≥ 28672 |
| 512 | 128 | 1024 | ≥ 26624 |
| 576 | — | 2048 | ≥ 24576 |

`FLASH_MLA_DISABLE_DSA_MLS_PREFILL` 关闭、`FLASH_MLA_FORCE_DSA_MLS_PREFILL`
强制（后者只查 `can_run`）。`can_run` 硬约束：d_v=512、d_qk∈{512,576}、
h_kv=1、h_q∈{64,128}、topk≤1024 或 2048（2048 时禁 attn_sink/topk_length）。

kernel 参数（params_adapter.h）：`seqlen_q = s_q * ngroups`（h_q 以 64-head
组折叠进序列维，`seqlenq_ngroups_swapped=true`）、`partition_size = topk`、
`num_splits=0` 表示 prefill。BOOL_SWITCH 组合：Is_MTP（mtp>1）、Is_causal、
Has_extra、DecodeCLoad。h_q=128 表示为两个连续 64-head 组。

## BF16 DSA sparse decode（sparse_bf16_dsa/fwd.cu）

- gfx936 走"prefill kernel 复用"路径（`run_gfx936_prefill_kernel_decode`）：
  h_q∈{32,64,128}（h_q=32 展平 batch 进 s_q、强制 splits=1）、d_qk=512；
  KV cache 字节数 > 4 GiB 时切 `decode_use_c_load`。
- gfx938 走 DSA kernel：h_q∈{64,128}、d_qk=512；`decode_use_c_load` 阈值
  `padded_k_tokens > 32*64*1024`（约 2M token）。
- **split 数启发式**（`default_num_splits`，行 61-138）：按
  `decode_tasks = b*s_q`、topk、extra_topk、h_q、c_load 分档返回
  32/16/8/4/2/1——纯经验表，覆盖 extra_topk>0、topk<512/512/1024/>1024
  各分支。`FLASH_MLA_BF16_DSA_FORCE_SPLITS` 可强制（调参入口）。
- split>1 时 partial 写 fp32 `split_o_accum/split_lse_accum`，由
  `gfx9::decode::bf16_dsa::run_flash_mla_combine_kernel` 合并（combine 支持
  attn_sink）。

## 与 GLM-5.3-Flash / vllm-plugin-das 的关系

- vllm-plugin-das 有两处绑定：`patch_sparse_mla_attention.py`（把 vLLM
  `SparseMLACommonImpl.do_kv_cache_update` 换成 `hcu_ops.concat_and_cache_mla`）
  与 `patch_flashmla_sparse.py`（替换 vLLM `FlashMLASparseImpl` 的
  fp8/bf16 kernel 调用与 MetadataBuilder）。GLM5Next 的 DSA 层经由
  sparse MLA 通用接口消费这些 kernel。
- 注意仓库无 `glm5` 命名模型文件；GLM 侧接入全部通过 vLLM 的
  `vllm.model_executor.layers.attention.sparse_mla_attention` 通用层 +
  HCU backend registry 间接完成。
- FP8 sparse decode 的 `splitkv_mla.cuh`（810 行）按 `ModelType::V32`
  （d_k=576、4 段 scale）与 `MODEL1`（d_k=512、7+1 段 scale）分模板；
  persistent 形态实例 `model1/v32_persistent_h{16,64,128}`。本轮未逐行精读，
  机制仅从 config.h 与 dispatch 层确认。

## 证据限制

- 本页为 source-level 阅读；**没有**本机 gfx938 上的编译、数值对照或计时。
- `prefill_kernel_opt_readme` 的 47-64% 延迟数字是上游在 gfx936 上的自报
  （source-reported），且该 readme 描述的 fast path 与本次阅读的 main 分支
  dispatch 收窄策略一致但细节（如 IS_TOPK_2048 窗口）未做逐条核对。
- legacy `flash_fwd_b16_mla.h`（DSA MLS 实现）与 `splitkv_mla.cuh`（FP8
  decode kernel 体）未逐行精读；本页对它们的描述限于 dispatch/参数层。
- 上游 main 与部署镜像版本可能不同（参照 vllm-plugin-das 的既有发现），
  使用前应以实际部署 SHA 为准。
