---
id: doc-hygon-moe-aiter-backend
title: 海光 Moe（moe_c）：MoE GEMM 手写 intrinsic 后端的源码契约
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- moe
- precision
- runtime-dispatch
confidence: source-reported
date: '2026-10-30'
captured_at: '2026-10-30'
url: https://github.com/HYGON-AI/Moe/tree/main/csrc_for_aiter
source_commit: 46ae5db
---

## 审阅范围

本页记录对 `Moe` 仓库（部署名 `moe_c`）的直接源码阅读，锁定 commit
`46ae5db`（2026-10-19，main）。README 未声明外部上游（Hygon 自研、
面向 DCU 的 MOE 算子库，API 与 AITER MOE 调用路径兼容）。审阅等级：
direct-source-read（README + 精度矩阵 + intrinsic 层 + w8a8 家族
主循环 + MODE 编码/dispatch 链 + 调优脚本面 + aiter 集成边界）。
本页**没有**本机编译或运行。

阅读的文件：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `README.md` | — | 精度矩阵、目录、aiter 集成流程 |
| `csrc_for_aiter/936_938/intrinsic_2.h` | — | mmac/buffer_load_lds 封装 |
| `csrc_for_aiter/936_938/intrinsic.h` | — | tcp_cache_swizzle SRD 构造、MATRIX_LOAD/DS_READ 宏 |
| `csrc_for_aiter/936_938/moe_w8a8_utils.h` | 4149 | w8a8 GEMM 主循环（精读一段） |
| `csrc_for_aiter/936_938/moe_w8a8_config.h` | — | kernel map 注册表 + MOE_W8A8_MODE 宏 |
| `csrc_for_aiter/936_938/`（目录） | ~50000 | 45 个头文件的家族清单 |
| `csrc/moe_c_w8a8_fp8.cu` / `moe_c_common.h` | — | torch 注册 + mode 解码 + kernel map 查找 |
| `moe_utils/fused_moe_c.py` | ~2400 | Python dispatch、量化、config 加载 |
| `moe_utils/moe_ops.py` / `ck_sorting_sim.py` | — | moe_sorting_fwd 绑定与模拟 |
| `moe_c_configs/gfx938/` | — | 按 quant 组织的 tuned config JSON |
| `tune/`（README_hipc.md 等 9 个脚本） | — | 独立 HIPC GEMM 调优器 |
| `aiter-das/aiter算子库说明.md`（跨仓库） | — | moe_c 后端在 aiter_moe 里的位置 |

## 精度矩阵与家族

W4A8(channelwise)、W8A8-INT8(channel/block/tensorwise)、
W8A8-FP8(channel/block/tensorwise)、W4A16(groupwise)、
W8A16(channel/block)、WFP4A8(channel/groupwise32)、WFP4A16
(groupwise)、W16A16(channel)；配套 top-k softmax、moe align/sum、
silu-and-mul。`csrc_for_aiter/936_938/` 单目录 45 个头共 ~50000
行，最大单文件 `moe_wfp4a8_utils.h` 5848 行。**架构目录名
`936_938` 即声明 936/938 双架构兼容**（intrinsic 守卫同时含
`__gfx936__ || __gfx928__ || __gfx92a__ || __gfx938__`）。

## intrinsic 层（与已知仓库的交叉验证）

`intrinsic_2.h` / `intrinsic.h` 的封装：

- **MMAC**：`mmac<int8_t>` 走
  `__builtin_hcu_mmac_i32_16x16x32_i8`（936/928/92a/938）；FP8 走
  `__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(v1,v2,v3, 0, 0)`
  与 `_lit` 变体（第 4 参 LIT 位）——**仅 gfx938**，非 938 返回
  零向量（编译不炸、运行为零）。这是 LIT/LTS 语义的又一消费者
  （对照 [[doc-hygon-hytlass-templates]] 的 ABI 注释与
  [[doc-hygon-ck-templates]] 的 quant 门控）。
- **global→LDS 直达**：`__builtin_hcu_raw_buffer_load_lds(rsrc,
  lds_ptr, dwords, v_offset, s_offset, imm, aux)`，aux bit3 是
  cache swizzle 开关；旁边保留被注释掉的
  `buffer_load_dword ..., lds` asm 原型（含 `s_mov_b32 m0`）。
  与 triton-das 的 BufferLds（async copy 落 LDS）、TileKernels 的
  buffer→smem 拷贝是同一硬件能力的三条使用路径。
- **LDS 矩阵读**：`ds_read_matrix_trans_format` 宏里
  `s_add_u32 m0, %1, 0x80000000`——**soffset bit31 编码 trans**，
  与 CK 的 `CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG=0x80000000`
  完全一致（本 wiki 现有两处独立佐证该 ABI 位）。
  `MATRIX_LOAD_64x32_B8_LDS_REARRANGE` 用两条
  `matrix_load_64x16_b8` 拼 64x32。
- **tcp_cache_swizzle_func_b8<stride>**：构造 buffer 描述符
  （res[1] += 0x40800000 即 SRD bit62 cache swizzle + stride 域，
  res[2]=0x80000000、res[3]=0x00020000）。注释保留了 256/196/128/64
  四档 stride 的编码表（现版本硬编码 128 档）。与 CK WASP pipeline
  的 CACHE_SWIZZLE VL1-Tag stall 修复同族。

## w8a8 GEMM 主循环（moe_w8a8_utils.h 精读段）

模板参数即 tile 词汇（`moe_c_w8a8_fp8.cu` 头注释）：
BLOCK_MNK 16×128×128、WARP_MNK 16×32×64、MMA 16×16×32。结构：

- token 行索引先 `inline_buffer_load_dword` 预取（sorted_token_ids
  压缩编码：低 24 bit token id + 高 8 bit topk id），`vmcnt_only_wait(0)`
  后展开为 A 行地址；
- A/B 走 `inline_buffer_load_dword_lds` 双 stage 预取（stage 偏移
  `warp_k_num * WARP_K`；注释：GEMM2 WarpK=64、stage 最多 2、
  smem 8K、"不解 bank conflict"）；
- B 权重按 marlin 式预 shuffle（`g_row_B` 的
  `(row_id&7)*32 + row_id/8*1024 + col_id*256` 位交织）；
- 输出经 `out_smem`（extern shared）做 warp 间 K 归约重组。

LDS 复用同步是活跃工程问题：`tune/test_lds_reuse_sync.py` 专门
"Repeat independent GEMMs to check LDS reuse synchronization"，用
`torch.allclose` 捕获输出 buffer 交替验证——说明独立连发 GEMM 时
LDS 内容复用存在已知的正确性风险（负面结果保留）。

## MODE 编码与三层 dispatch

`tune/` 的调优器围绕一个十进制位域展开（README_hipc.md +
`MOE_W8A8_MODE(STAGE, SORTING, FAMILY, BM, PROFILE)` 宏）：
`STAGE×10^7`（1=gemm1/2=gemm2）、`SORTING×10^6`（0=CK sorting/
1=moe_align）、`FAMILY×10^5`（0=decode/1=prefill）、
`(BM/16)×10^3`、`PROFILE`（nloop/stage 变体，101-105 等）。

例如配置文件里的 `20001103` = gemm2 + CK sorting + prefill +
BM16 + profile 103。三层：

1. **Python**（`fused_moe_c.py`）：按 M/N/K/topk/cu 从
   `moe_c_configs/gfx938/<quant>/` JSON 选 MODE+BLOCK_SIZE_M
   （文件名即 shape key，如 `E=160,N=320,K=6144,topk=8,cu=72`）；
   校验 `BLOCK_SIZE_M == mode//1000%100*16`。
2. **C++ 注册层**（`moe_c_common.h::dispatch_gemm_stages`）：
   解码 mode（`is_decode = (mode/100000)%10==0` 等），按
   dtype/stage/sorting 选 8 张 `kernel_maps_*` hash 表之一；
   INT8 强制 CK sorting，W4A8 强制 moe_align，FP8 两者皆可；
   查不到即抛错（无静默回退）。
3. **kernel map**（`moe_w8a8_config.h`）：mode → 具体模板实例
   lambda，如 `MOE_W8A8_MODE(1,1,1,32,102)` →
   `launch_..._prefill_fp8_nloop<32,128,64,32,32,64,2,2>`。

调优器（tune_moe_hipc.py 家族）独立生成 GEMM1/GEMM2 输入、逐
expert torch 矩乘做参考、路由/打包/量化全部移出计时区间、每个
候选连同正确性与最大绝对差写 CSV、按 GEMM1+GEMM2 延迟和选组、
同 BM 同 sorting 才可配对。

## aiter 集成边界（跨仓库核对）

- aiter-das `.gitmodules` pin `3rdparty/moe_c → ../Moe.git`
  @ `9075f00`（另 pin composable_kernel @ a870f1b）；本仓库 HEAD
  `46ae5db` 与该 pin 存在距离（社区演进 vs 部署快照，同
  TileKernels/vllm-plugin 的模式）。
- aiter `aiter_moe` 三路径：**moe_c 后端**（本仓库，marlin 式
  GEMM1→silu→GEMM2→sum）、ASM 后端（`AITER_ASM_DIR` 下的预编
  asm，经 `get_moe_asm_solution` 选 solution）、Triton 后端。
- `moe_utils.moe_ops.moe_sorting_fwd` 在本仓库内是 **Python 模拟**
  （`simulate_ck_moe_sorting`，torch ops 拼出排序结果后 copy_ 进
  输出 buffer）——真实 sorting kernel 在 aiter/CK 侧；本仓库
  单独跑 moe_c GEMM 时用模拟器自洽。
- vllm-plugin-das 对 moe_c 无直接 import（grep 无）；其 MoE 路径
  经 aiter 间接使用。

## 对本机优化工作的含义

- 这是 GLM-5.3-Flash MoE 层（GLM5Next 的 MoE block）最直接的
  **社区手写对照实现**：aiter-das 已收录的 FLA 页只覆盖 attention
  侧，本页补上 MoE GEMM 侧。W8A8-FP8/W4A16 等 quant 路径的 tuned
  config（gfx938 目录）是现成的 shape→tile 起点表。
- MODE 位域 + kernel map 的 dispatch 设计（无静默回退、INT8/W4A8
  的 sorting 约束写成 TORCH_CHECK）是 runtime-dispatch 模式的
  良好参照。
- 三个负面/风险记录值得引用：独立 GEMM 连发时 LDS 复用同步风险
  （tune 测试脚本专门验证）；FP8 mmac 在非 938 架构返回零向量
  （静默数值错误面）；gate+up fp16 累加器溢出（CK fused_moe 同样
  记录）。
- `tcp_cache_swizzle` 的 SRD 编码注释（stride 档位表）是手工构造
  cache swizzle 描述符的少数公开参考。

## 证据限制

- 本页为 source-level 阅读；**没有**本机编译或运行；README 与
  tune/ 里的任何性能声明均为 source-reported。
- 45 个头只精读了 intrinsic 层与 `moe_w8a8_utils.h` 的
  decode/nloop 段；wfp4a8（5848 行）、w4a16_2、prefill 专用
  （bk64/bk128）等家族只做清单。
- `moe_sorting_fwd` 的 Python 模拟器只确认了存在与用途，未逐行
  核对排序语义与 CK sorting kernel 的一致性。
- aiter pin（9075f00）与本仓库 HEAD 的差异内容未逐 commit 比对。
