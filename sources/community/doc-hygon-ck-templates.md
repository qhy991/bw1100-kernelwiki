---
id: doc-hygon-ck-templates
title: 海光 composable_kernel-das：CK/ck_tile 的 HCU 适配层（MLS atom、WASP pipeline、abarrier）
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- gemm
- attention
- moe
confidence: source-reported
date: '2026-10-30'
captured_at: '2026-10-30'
url: https://github.com/HYGON-AI/composable_kernel-das/tree/main/include
source_commit: e295c92
---

## 审阅范围

本页记录对 `composable_kernel-das` 仓库的直接源码阅读，锁定 commit
`e295c92`（2026-10-19，main）。上游为 ROCm `composable_kernel`
（commit `4feebedd`）。审阅策略：**只读 HCU 专属层**（ck utility 的
hcu 头 + ck_tile arch 层 hcu 头，共 ~3900 行）+ WASP grouped-GEMM
pipeline 头注释精读 + `example_hcu/` 清单与关键 README + 两份仓库级
中文说明文档；24M 的通用模板层（`include/ck` 其余部分）不精读。
本页**没有**本机编译或运行。

阅读的文件：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `include/ck/utility/hcu_mmac.hpp` | 176 | ck(2.x) 层 MMAC intrinsic 统一封装（14 个） |
| `include/ck/utility/hcu_dsreadm.hpp` | 168 | ds_read_m intrinsic 封装（14 个） |
| `include/ck_tile/core/arch/hcu_mls_atom_gfx938.hpp` | 942 | gfx938 MLS atom（12 个，裸 asm） |
| `include/ck_tile/core/arch/hcu_mls_traits_gfx938.hpp` | 262 | MLS traits（PackedShape） |
| `include/ck_tile/core/arch/hcu_abarrier.hpp` / `hcu_ebarrier.hpp` | 187/116 | 到达/发射屏障 asm 封装 |
| `include/ck_tile/core/arch/hcu_buffer_addressing.hpp` | 356 | SRD 构造（含 cache swizzle） |
| `include/ck_tile/core/arch/hcu_matrix_addressing.hpp` | 910 | m0/matrix_load 地址层 |
| `include/ck_tile/core/arch/wasp_helper.hpp` | 41 | warp 特化下的 warp id 抽象 |
| `include/ck_tile/core/arch/hcu_quant_gemm_target.hpp` | 62 | 938/946 quant GEMM 能力门控 |
| `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_wasp.hpp` | ~1500 | WASP grouped-GEMM pipeline（头注释即设计文档） |
| `include/ck_tile/core/config.hpp`（选段） | — | arch 宏门控 |
| `example_hcu/`（清单） | — | 17 个传统示例 + ck_tile 30+ 个 |
| `example_hcu/ck_tile/{19_grouped_gemm,23_unified_attention,17_fused_moe,20_jenga}/README.md` | — | 示例文档 |
| `ck_optimize.md` / `composable_kernel算子库说明.md` | — | 迁移与优化工作流记录 |

## 定位：第三条路线（模板库 + 显式 pipeline）

本仓库与已收录仓库的关系：

- 与 **hytlass**（[[doc-hygon-hytlass-templates]]）同为模板库路线，但
  ck_tile 多了 **HCU 专属 pipeline 词汇**（MLS/WASP/abarrier 进
  pipeline 抽象层），且 HCU 适配深度更大（hytlass 的 gfx938 atom 只有
  2 个 FP8；CK 的 MLS atom 有 12 个 b16/b8）。
- 与 **triton-das**（[[doc-hygon-triton-hcu-backend]]）共享同一组
  概念（MLS、WASP、BufferLds、abarrier），但实现层不同：triton 在
  MLIR pass 里自动选择，CK 在 C++ 模板/pipeline 层显式组合。两仓库
  可互为语义参考。
- 与 **aiter-das**：aiter 把 composable_kernel pin 为 submodule
 （`3rdparty/composable_kernel` @ a870f1b，同时 pin Moe @ 9075f00，
  见 [[doc-hygon-moe-aiter-backend]]）。

## HCU 专属层事实

### ck(2.x) 层：intrinsic 统一封装

`hcu_mmac.hpp` 14 个 `intrin_mmac_*` 模板，arch 守卫写法与 hytlass
`mmac_gfx9.h` 同构但**用旧式非 LIT/LTS builtin**（如
`__builtin_hcu_mmac_f32_16x16x8_tf32`），差异点：

- fp64 vstep 变体（`__builtin_hcu_mmac_16x16x4_f64_vstep`，仅
  936/938）——hytlass 没有这一条。
- i4/u4（16x16x64）在 936/938 有 builtin；FP8 在 92a/938/946 有，
  其中 f8_f8 混合组合（bf8_f8 等）**仅 92a/938**。
- `hcu_dsreadm.hpp`：`ds_read_m32x8_f32/_tf32`、`m32x16_f16/_alt`、
  bf16、`m32x32_u8` 等 14 个，arch 守卫覆盖 928/92a/936/938/946——
  与 hytlass `memory_gfx928.h`、triton `DsReadM.cpp`、deepgemm
  `DS_READ_MATRIX_*` 宏同一族（注意这里多了 **tf32 的 m32x8** 与
  **f16 的 m32x16** 变体）。

### ck_tile 层：MLS atom（gfx938 专属 12 个）

`hcu_mls_atom_gfx938.hpp` 以**裸内联汇编**封装
`matrix_load_32x16_b16 / 32x32_b16 / 64x16_b16 / 64x16_b8 / 64x32_b8 /
128x16_b8`（及 trans 变体），关键细节：

- `s_mov_b32 m0` + `s_nop 0` 的发射模式与 FlashMLA 的 LDS 预取
  写法一致（`use_m0`/`s_nop` 是模板开关）。
- **trans（转置）通过 soffset 的 bit31 编码**：
  `smem | CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG`，其中该宏在
  gfx938 下为 `0x80000000`、其他架构为 0（config.hpp:171-174）。
  这与 Moe 仓库 `ds_read_matrix_trans_format` 前的
  `s_add_u32 m0, %1, 0x80000000`（[[doc-hygon-moe-aiter-backend]]）
  **是同一 ABI 细节的两处独立实现**——本 wiki 现在有三个来源交叉
  验证该 bit。
- traits（`hcu_mls_traits_gfx938.hpp`）只给 PackedShape 与 warp 级
  偏移；注释明确 "actual lds layout will always be wrapped in 16
  Dwords on major dim"，线程级偏移不做。
- gfx946 另有 `hcu_mls_atom_gfx946.hpp`（支持 BPS，
  `CK_TILE_MATRIX_LOAD_SUPPORT_BPS`）。

### abarrier/ebarrier（gfx946 独占）

`s_abarrier_init/arrive/try_wait/slot_rd` 等一整套标量屏障 asm
封装，**arch 宏只在 gfx946 打开**（`CK_TILE_DEVICE_ABARRIER_SUPPORT`
/`EBARRIER_SUPPORT` 都 gated on `__gfx946__`）。abarrier slot 的
位域（19 bit transaction count、11+11 arrival/expected count、
phase、valid）是源码里对硬件屏障语义最具体的描述。gfx938 上这些
指令不可用——这解释了 triton-das 里 WASP 运行时测试为何只测 946。

### quant GEMM 能力门控

`hcu_quant_gemm_target.hpp`：host 编译时靠 CMake 传
`CK_TILE_HCU_QUANT_GEMM_GFX*`（编译器 arch 宏只在 device pass 可
靠）；**gfx938 上 FP4/INT4/MXFP8 被禁用**
（`CK_TILE_ARCH_SUPPORTS_FP4 = !GFX938_TARGET`）且强制 OCP FP8
格式。这是 "938 无 FP4 硬件路径" 的又一独立佐证（与 triton-das
MlsGroup 表 938 段无 b4 条目一致）。

## WASP grouped-GEMM pipeline（设计文档级头注释）

`gemm_pipeline_ag_bg_cr_wasp.hpp` 的头注释是本仓库最有价值的工程
记录，摘录关键实测结论（均 source-reported，shape 为
256M×256N×64K grouped GEMM）：

- **结构**：12-wave workgroup（4 producer + 8 consumer wave，
  768 线程）；wave grid 只描述 consumer（1×8，warp tile 16×16×16，
  32 MMAC、128 acc VGPR/lane）；LDS 64KiB 手管，2×32KiB A ring，
  epilogue 复用同一 buffer。
- **VALU 消除**：k 循环按构造零 VALU——每 lane 地址按 tile 构造一
  次，stage 变化量走 SGPR soffset 或 ds_read 的立即数域。实测
  consumer 31→0、producer 42.5→0 条 VALU，**245.6→285.0 TFLOPS，
  VGPR 254→241，零 spill**。额外难点是让 hoisted 值对 LLVM 重物化
  启发式"不可复活"——它偏好循环内重建廉价不变量而非占寄存器。
- **B 流读的 CACHE_SWIZZLE**：裸指针解引用在 B 流上遭遇 VL1-Tag
  RAM stall（整 wave 同 tag bank）；改用 buffer_load_dwordx4 +
  CACHE_SWIZZLE 描述符（SRD 的 STRIDE 即 tag-bank 旋转周期）解决。
  四个 load 合一个 asm 块、K-half 差走立即数 +64B。
- **四分之一相错（C/C' persona）**：consumer 前后两半组旋转 Q3
  相位。历史 cadence 实测：C' 序 +4~7%，中段 rendezvous -12%，
  rotation 本身再 -35%——**rotation 是唯一净亏的配置**，且亏损
  不是 codegen（spill 三行相同）。寄存器侧：整 K-half fragment
  （64 VGPR）跨 barrier 存活会顶到 256-VGPR 上限（spill 260），
  8-unit 分块（spill 5/24B）与参考 kernel（也 spill，1/4 量）的
  对照都在注释里。
- 参考实现是外部 `bf16_gemm_nt_hipc.cpp`（不在本仓库），port 注释
  多处引用其行号——"hipc" 是该手写参考 kernel 的代号。

## example_hcu 清单（选摘）

- **19_grouped_gemm**：fast/smoke target 体系；gfx938 quant
  grouped GEMM（FP8/BF8 四输入组合 × FP16/BF16 输出 × NN/NT/TN
  75 用例）；gfx936/938 对齐 shape 用 4×2 waves、512 线程的
  spill-reduction 版本；variable-K（组间 K 不同）非 split-K。
- **23_unified_attention**：2D/3D unified attention 公共算子
  （D192/D256、paged KV、sliding window、attention sinks、ALiBi、
  softcap、PrefixLM；3D 按 softmax segment 发射 + reduce 合并，
  sink 保留在首个有效 partial）。README 带 gfx936 benchmark 表
  （MIMO/Qwen 用例）。
- **17_fused_moe**：moe-sorting + group-GEMM 两段融合（自称比
  vllm 方案 1.5~2x，source-reported；gate+up fp16 会累加器溢出，
  强制 BF16）。
- **20_jenga**：block-sparse attention FWD/BWD（64×64 稀疏块，
  head dim 128）。
- **38_block_scale_gemm**：gemm_quant tensor/rowcol/blockwise 三种。
- 传统 `example_hcu/`（非 ck_tile）：01-03 grouped conv
  fwd/bwd（mmac 版）、05/06 GEMM、15/24/25 grouped/batched GEMM、
  26 mmac fp32/fp64（README 给了 gfx936/938 手动 hipcc 命令）。

## 迁移工作流记录（ck_optimize.md）

仓库自带的 ck_tile HCU 迁移 skill：从最接近的
`example_hcu/ck_tile` 实现起步；改代码前先记录
dtype/shape/验证状态/延迟/吞吐/GPU id/编译器；benchmark 前
`rocm-smi --showuse --showmemuse` 选低负载卡 +
`HIP_VISIBLE_DEVICES`；一次只加一个 `-config=` 候选，同场基准打
赢当前冠军并通过验证才转正；**"HCU grouped GEMM 用 MMAC 不用
MFMA，加新指令路径前先搜 `include/ck_tile` 附近现成的 MMAC 别名
/MLS shape/inline asm"**；fp16 性能在正确性修复前不算数（同
shape 族在 03_gemm 同样出错）。

## 对本机优化工作的含义

- CK 的 MLS atom + trans bit31 + m0 发射模式，与 FlashMLA/Moe/
  triton 的对应物构成同一 ABI 的**四路独立实现**，是手写 kernel
  对齐 LDS 读布局时最可靠的交叉验证组。
- WASP pipeline 头注释记录的三个实测结论（VALU 零化 +16% 吞吐、
  B 流 CACHE_SWIZZLE 解 VL1-Tag stall、C' rotation 净亏）是社区里
  最详细的 warp 特化工程数据，且明确标注了历史测量与当前
  Q0-Q3 排程的区别——引用时注意区分。
- abarrier 仅 gfx946：gfx938 上的 warp 特化同步仍走传统
  barrier（与 triton-das gfx938 WASP 运行时覆盖弱的状态互证）。
- `example_hcu/ck_tile` 是本机可编译的现成算子面（attention/
  grouped GEMM/MoE/quant GEMM/jenga），若 KDA 选定 MoE 或 attention
  选题，可作社区对照实现；其 README benchmark 表是 gfx936 数字，
  BW1100 上需本机重测。

## 证据限制

- 本页为 source-level 阅读；**没有**本机编译或运行。WASP pipeline
  注释里的 TFLOPS/spill 数字与各 README 的性能声明均为
  source-reported，且注明测量环境为特定 shape 与（多数）gfx936。
- 通用模板层（`include/ck` 的 tensor_operation/device operator 层、
  profiler、python/ck4inductor）未精读。
- `bf16_gemm_nt_hipc.cpp` 参考实现不在仓库内，其行号引用无法核对。
- MLS atom 的 r 标志（readable/returned register 语义）与
  `s_nop 0` 的时序要求只按 asm 形式记录，未对照 ISA 文档。
