---
id: doc-hygon-triton-hcu-backend
title: 海光 triton-das：Triton release/3.6.x 的 HCU 后端（MLS/WASP/BufferLds/ScaleBuffer）
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- triton
- gemm
- compiler-coevolution
- precision
confidence: source-reported
date: '2026-10-30'
captured_at: '2026-10-30'
url: https://github.com/HYGON-AI/triton-das/tree/main/third_party/hcu
source_commit: 82bd651
---

## 审阅范围

本页记录对 `triton-das` 仓库的直接源码阅读，锁定 commit `82bd651`
（2026-10-22，main）。上游为 Triton `release/3.6.x` 分支
（基线 commit `85400f80bf`），HCU 适配集中在 `third_party/hcu/`
（backend + lib + test 三处），架构支持
gfx926/928/936/938/92a/946。审阅等级：direct-source-read
（backend 选项面 + MLIR pass 层抽读 + LDS 读模型头文件 + 测试面清点）。
本页**没有**本机编译或运行。

阅读的文件（`third_party/hcu/` 下）：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `README.HCU.md` | — | 上游基线与适配范围声明 |
| `backend/compiler.py` | ~700 | arch 参数、HIPOptions、pass 管线装配 |
| `backend/wasp.py` | — | WASP 选项校验 |
| `python/triton/language/core.py:2173` | — | `tl.matrix_load` builtin 定义 |
| `python/triton/language/semantic.py:1109` | — | matrix_load 语义（2 维限定、mask 软件化） |
| `lib/TritonHCUTransforms/MlsGroup.cpp` | 542 | MLS 指令选择表（gfx938/92a/946 三段） |
| `lib/TritonHCUTransforms/MlsLowering.cpp` | 250 | matrix_load → local_alloc/async_copy/local_load |
| `lib/TritonHCUTransforms/MlsStreamPipeline.cpp` | 1250 | MLS 软件流水（modulo schedule） |
| `lib/TritonHCUToLLVM/DsReadM.cpp` | 659 | ds_read_m / BufferLds 到 LLVM 的降级 |
| `include/TritonHCU/LdsReadModel.h` / `LdsReadAccessPatterns.h` | — | LDS 矩阵读的 bank 冲突模型与 lane 相位表 |
| `lib/TritonHCUTransforms/WaspSplitPlan.cpp` / `PrepareWaspSplitPlan.cpp` | — | WASP 拓扑与 split 计划 |
| `lib/TritonHCUTransforms/ScalePipeline.cpp` / `TritonHCUToLLVM/AllocateScaleBuffer.cpp` | — | dot_scaled 的 Scale Buffer 流水 |
| `lib/TritonHCUTransforms/SelectBufferLdsConfig.cpp` | — | bank 冲突感知 BufferLds 编码选择 |
| `test/`（66 个 .py） | — | 测试与微基准面 |

## 四个 HCU 专属优化机制

### 1. MLS（matrix_load / matrix_store）

Python 层新增 `tl.matrix_load` / `tl.matrix_store` builtin
（`core.py:2173`，语义层限定 2 维；`matrix_load` 本身不支持原生
mask，semantic 层用 `tl.where` 软件补偿）。用户用它声明
"A/B 操作数直接以 MMAC 消费布局落到 LDS"，随后三段降级：

1. `MlsEncodingInsertion`（701 行）给 matrix_load 结果打
   `MlsEncodingAttr`；
2. `MlsLowering` 把
   `matrix_load → local_alloc + matrix_load_to_local + async_commit +
   async_wait + local_load`（或 `local_load_packed_mls_shared`）；
3. `MlsStreamPipeline`（1250 行）做 modulo 软件流水——注释明确
   "保留旧 triton stream-prefetch 语义：global load 与 LDS 消费
   隔一个流水级"（compiler.py 注释）。

指令选择表在 `MlsGroup.cpp`：按
`(nonKTile, kTile, transpose, elemBitWidth, elemBitTyKind, alt2Kind, version)`
七元组键查 `hcu_matrix_load_32X16/64X16/128X16/256X16_b16/b4` 与
`hcu_ds_read_matrix_format/_trans_format/_padbyte` 的组合，并给出
逐指令的 LDS 字节偏移表。分三段：gfx938（b16/b8 为主）、
gfx92a+gfx946、gfx946 专属（b4/b4mix packed，b4 明确 TODO
"add i4, u4 if need"）。测试面 `mls-unit-test.py` 提供
`mix_load` 掩码（0b00 双 MLS / 0b01 仅 A / 0b10 仅 B）对照，
`is_hcu_support_mls()` 要求 gfx938/946/92a。

### 2. WASP（warp specialization）

- 拓扑：`wasp_partition_warps` 元组（默认 `(4,4,4)` = 1P2C），
  长度选 1P1C/1P2C/2P2C；每分区 1/2/4/8 wave，总 wave ≤16；
  **开 WDRA（`wasp_wdra=True` 默认）时每个分区必须恰好 4 wave**。
  Producer（Load）在前、Consumer（MMA）在后，Consumer wave 数
  ≥ Producer。
- `PrepareWaspSplitPlan` 在布局选择前把 split 意图记到 dot 上
  （先按 M/A 维、再 N/B 维切，要求 extent ≥ factor×16 且可整除）；
  用户显式选了拓扑但 CTA tile 切不开时**编译期报错**，不静默
  退回 1P1C。
- 运行时测试 `test_wasp_runtime.py` 要求 gfx946（否则 skip）；
  gfx938 上 WASP 属编译可开但运行时覆盖弱的状态。
- 开 WASP 时跳过 MLS 软件流水与 amd stream_pipeline
  （"WASP owns multi-buffering / sync"）。

### 3. BufferLds（bank 冲突感知的 buffer→LDS 放置）

`SelectBufferLdsConfig` 在最终 dot-operand 编码确定后，把普通
async-copy 尝试替换为带 bank 冲突代价模型的共享编码；不合法的
操作数保持原样，一个都没选上则整体回退通用 async 管线
（`uses_bank_conflict_aware_lds_placement` 元数据记录结果）。
选中的操作数走 `DsReadM.cpp` 降级，落到
`__builtin_hcu_ds_read_m32x8f32 / m32x16f16(_alt) / m32x32u8`
（与 hytlass `memory_gfx928.h`、FlashMLA LDS swizzle 同一族原语）。

核心数据结构是 `LdsReadAccessPatterns.h` 里的 **lane 相位表**：
b16 读的 8 组 `lane ^ shift`（shift ∈ {0,2,8,10,32,34,40,42}）、
b8 读的 8 组（shift ∈ {0,1,8,9,32,33,40,41}）、b32 的 B128 表。
注释说明这些 lane 组是 **gfx936/gfx938 上 pair-collision 探针实测
验证**的，92a/946 "share that profile"，且强调"不蕴含相位 issue
顺序"；b4 被刻意排除——实测的冲突周期增量不同，仅 lane 划分尚不
构成充分代价模型。配套探针工具 `probe_lds_phase_pairs.py` /
`run_lds_bank_probe.py`：差分 pair-collision 实验（对照读 4..7 vs
8..11 bank，碰撞读 4..7 内不同 word），用
`xprof --sections memory_workload` 的 LDS-Bank Conflict 计数器差值
作观测量。目标矩阵读按 arch 门控：gfx936 全支持到 M32x8B32，
gfx92a/938/946 只建模 b16/b8 两种 opcode（`hasLegacyMatrixReadProfile`）。

### 4. Scale Buffer（dot_scaled / FP8 FP4）

`AllocateScaleBuffer` 把 dot_scaled 的 scale 张量放进 HCU 专属
Scale Buffer（独立于 LDS 的片上存储，512 行 × 16 列 × 1B = 8KiB
= 32 slot；M16/N16 MMAC group 内 col 选 M/N、row 选 K32 组，
E8M0 scale 每 32 个输入元素一个）。`ScalePipeline` 负责软件流水，
gfx946 额外可用原生 Global→LDS 拷贝。测试面
`test_scaled_compile.py` 校验 `@llvm.hcu.mmac.scale.` intrinsic
生成，`mls-dot-scaled-test.py` 做 MLS 与 scaled 组合。

## backend 选项面（compiler.py）

- **768 VGPR**：gfx936/938 传
  `-mllvm=-support-768-vgprs=true`（928/946/92a 为 512）。
- **架构参数**：gfx936/938 共用一组（`arch in ("gfx92a","gfx936","gfx938","gfx946")`
  判定族），gfx946 有 `hcu_use_gfx946_sched_model_v2` 调度模型开关
  （autotune 可关掉以保留旧行为）。
- **BlockPingpong**：HCU-only 的 BufferLoad 三段流水匹配器，只接受
  实测过的 8-warp GEMM tile、合法 BK 减半、向量化 A/B load、LDS
  ≤64KiB；要求关 async copy 与 WASP；reduction 数据流后期选
  split-BK 四相或 whole-BK 两相体。`use_block_pingpong=False` 默认。
- **UTCWarmup**：大而规则的 GEMM 类 reduction 的 UTCL1 页预热，
  autotune-only，默认关（普通/小 kernel 会回退）。
- `buffer_cache_swizzle` 注释明确 "mls ignore this"（MLS 自带
  更优默认）。
- `-mllvm=-disable-cluster-lds-memops=true`、WDRA 时的
  `-vgpr-greedy-alloc-mode=local-wave`、`PMD_PATH` 存在时跳过
  backoff-barrier 参数——这组 LLVM 后端旋钮均在本文件装配。

## 测试面（66 个 .py）

正确性（`test_*.py`）+ 微基准（小写连字名：`matmul.py`、`fa.py`、
`gemm-ds-read-m.py`、`persistent-matmul.py`、`matmul_mls_wasp.py`
等）双轨；`lds-read-model.md` 状态文档声明 LDS 消费者模型仅在
gfx936 上做过硬件验证。这批微基准是社区里最接近
"KDA 可直接对照的 Triton 基线"的语料。

## 对本机优化工作的含义

- **编译器协同是这批社区仓库里独有的**：其余仓库（deepgemm/
  FlashMLA/lightop）是 kernel 侧手写，triton-das 是唯一暴露
  "MLIR pass + LLVM 后端旋钮"层面的仓库。若 KDA 的 Cake 编译
  侧要生成 HCU 代码，MLS/BufferLds/WASP 的选择逻辑与
  `LdsReadModel.h` 相位表是最直接的参考实现（另见
  technique-gfx938-instruction-audit 的独立审计可互验）。
- gfx938 上的注意点：WASP 运行时测试仅覆盖 gfx946；LDS 读模型
  硬件验证在 gfx936；gfx938 的矩阵读 profile 只建模 b16/b8——
  **从本机实测出发反哺这些表的空白**（如 b4 相位顺序、b32 在
  938 的 profile）是潜在的高价值实验方向（待与用户确认后再做，
  见 [[discuss-before-running-experiments]]）。
- `MlsGroup.cpp` 的 gfx938 段只有 b16/b8 条目、无 FP8/b4——
  FP8 scaled 路径在 938 上是否走 MLS 由 `mls-dot-scaled-test.py`
  的编译测试约束，运行时性能未在源码中声明。

## 证据限制

- 本页为 source-level 阅读；**没有**本机编译或运行。README/测试
  中的任何性能声明均为 source-reported。
- `MlsStreamPipeline.cpp`（1250 行）与 `AccelerateHCUMatmul.cpp`
  只读了头部与注释；流水调度细节以源码为准。
- LDS 相位表的"gfx936/gfx938 探针实测"结论引自
  `LdsReadAccessPatterns.h` 注释，探针数据本身（xprof 产物）不在
  仓库内，无法从本仓库复验。
- gfx92a/gfx946 段的指令表只做了覆盖面清点，未逐条核对偏移表。
