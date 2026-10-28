---
id: doc-hygon-hytlass-templates
title: 海光 HYTLASS：CUTLASS/hute 模板库与 gfx938 指令原语层的源码契约
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- gemm
- hip
- migration
confidence: source-reported
date: '2026-10-29'
captured_at: '2026-10-29'
url: https://github.com/HYGON-AI/hytlass/tree/release/include
source_commit: 47488c8
---

## 审阅范围

本页记录对 `hytlass` 仓库（release 分支）的直接源码阅读，锁定 commit
`47488c8`（2026-10-22，release 头部；首个公开版本 `4fad709`，即
FlashMLA-das 所 pin 的 submodule SHA）。上游为 NVIDIA CUTLASS `v3.5.0`
（commit `7d49e6c`），BSD-3。审阅等级：direct-source-read（arch 原语层 +
hute atom/traits 层 + CMake/工具链 + 一个教程示例）+ 目录清单。
本页**没有**本机编译或运行。

阅读的文件：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `include/hytlass/arch/mmac_gfx9.h` | 106 | gfx928/936/938 MMAC intrinsic 统一层 |
| `include/hute/arch/mma_gfx928.hpp` | ~860 | gfx928 族 MMA atom（42 个） |
| `include/hute/arch/mma_gfx938.hpp` | 124 | gfx938 专属 FP8 LIT atom（2 个） |
| `include/hute/atom/mma_traits_gfx928.hpp` | — | 63 个 MMA_Traits 特化 |
| `include/hute/atom/mma_traits_gfx938.hpp` | 74 | 2 个 FP8 traits |
| `include/hytlass/arch/memory_gfx928.h` | — | `ds_read_matrix` 家族 |
| `include/hute/arch/copy_gfx928.hpp` | 210 | DS_READ copy atom |
| `include/hute/arch/cluster.hpp` | 45 | cluster stub |
| `include/hytlass/float8.h` | — | FP8 使能开关 |
| `examples/hute/tutorial/sgemm_gfx928.cu` | 699 | hute 流水线 GEMM 教程 |
| `CMakeLists.txt`（行 160-180） | — | 架构列表逻辑 |

## 两层结构：hytlass(2.x) 与 hute(3.x/CuTe)

- `include/hytlass/`：CUTLASS 2.x 风格（threadblock/warp/epilogue 层次、
  SIMT 卷积、隐式卷积），支持 TF32/FP16/BF16/I8/U8。
- `include/hute/`：CUTLASS 3.x CuTe 模型的 HCU 移植（HuTe）——
  layout/swizzle/tensor/atom/traits 全套 CuTe 词汇。FlashMLA-das 的
  phase1 kernel 即建立在这层上（`hute::Tensor`、`TiledMMA`、
  `tile_to_shape`、`Swizzle<3,3,3>` 等）。

## gfx938 指令原语层（本页核心事实）

**统一 MMAC 层**（`mmac_gfx9.h`，gfx928/936/938 三架构共用）：

| 抽象 | gfx938 后端 | 旧架构后端 |
|---|---|---|
| `mmac_f32_16x16x8_f32` | `__builtin_hcu_mmac_16x16x8_f32_lit_lts(…, false, false)` | `__builtin_hcu_mmac_f32_16x16x8f32` |
| `mmac_f32_16x16x8_tf32` | `…_tf32_lit_lts(…, false, false)` | `…_16x16x8tf32` |
| `mmac_f32_16x16x16_f16/_bf16` | `…_lit_lts(…, false, false)` | `…_16x16x16f16/bf16` |
| `mmac_i32_16x16x32_i8/_u8` | `…_lit_clamp_lts(…, false, false, false)` | `…_16x16x32i8/u8` |

关键源码注释（行 35-36）：**"gfx938 instructions retain the gfx928/gfx936
lane layout when LIT, LTS and integer saturation are all disabled"**——即
gfx938 的 HCu 指令族通过关闭 LIT/LTS/饱和位与旧 ABI 寄存器布局兼容。
这条注释是所有跨 928/938 共享 atom 代码的合法性依据。i8/u8 在 gfx938 上
多一个 clamp 位；FP8 atom（见下）则真正打开 LIT。

**LDS 矩阵读**（`memory_gfx928.h` / `copy_gfx928.hpp`，同样三架构共用）：
`__builtin_hcu_ds_read_m32x8f32` / `m32x16f16`（+`_alt` 变体）/ `m32x32u8`。
deepgemm tf32 kernel 里的 `DS_READ_MATRIX_32X16_B16` 与 FlashMLA 的
LDS swizzle lane 映射，底层都是这组原语。

**gfx938 专属 hute atom**（`mma_gfx938.hpp`，仅 2 个，全是 FP8 E4M3 LIT）：

- `GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT`：一次
  `__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A, b, C, true, false)`
  调用对（A 复用、B 拆两半），LIT=true。
- `GFX938_16x64x64_…LIT`：8 次调用拼 64 宽 tile，A_0/A_1 交错。
- traits 给出 64 线程 (wave64) 的 A/B/C 寄存器→逻辑坐标 Layout。
- 单测 `test/unit/hute/gfx928/mma_composed.cu` 有 `GFX938_HuTe` 两个
  用例，非 gfx938 设备上 GTEST_SKIP。

**FP8 转换**：`float8.h` 只在 `__gfx938__ && __HIP_DEVICE_COMPILE__` 时
定义 `HIP_FP8_CVT_ENABLE 1`——FP8 数值路径是 938 独有的。

## 已知边界与不完善处（源码直读）

- **CMake 默认架构不含 938**：`CMakeLists.txt:173` 追加
  `906 926 908 928 936`；`hytlass_profiler` 的 generator 默认
  `--architectures '906;928;936'`。938 支持存在于头文件
  （`__gfx938__` 守卫），但默认构建目标与 profiler 生成器都没纳入——
  938 用户需用 `AMDGPU_TARGETS` 环境变量或显式传参。
  另外该 `elseif()` 分支条件为空（恒假），fallback 列表实际只在
  `AMDGPU_TARGETS` 未定义且……的路径上生效，属构建脚本的粗糙处。
- **cluster 是 stub**：`hute/arch/cluster.hpp` 的
  `block_id_in_cluster()` 直接返回 `{0,0,0}`——没有真实 cluster 支持，
  CuTe 3.x 里依赖 cluster 的模式（如 DSMEM 共享）在 HCU 上不可用。
- **gfx938 专属 atom 覆盖窄**：只有 FP8 E4M3 两个；bf16/f16 在 938 上
  复用 928 atom（靠 LIT/LTS 关闭位），这也是 FlashMLA
  `hytlass_ext.h` 里 `GFX928_16x16x32_F32BF16BF16F32_NT` 命名在 938
  设备上运行的原因。
- **hute 教程的流水线是 `__syncthreads()` 级**（sgemm_gfx928.cu 行
  263-310：gmem→rmem→smem→rmem 手工双缓冲），没有 CuTe 3.x 的
  Pipeline/smem descriptor 机制——async 深流水（如 deepgemm/FlashMLA
  里手写的 vmcnt/lgkmcnt 计数）超出该模板库的抽象范围。

## 消费方关系（跨仓库核对）

- **FlashMLA-das**：`.gitmodules` pin `hytlass@4fad709`（release 分支），
  无本地修改。`csrc/hytlass_ext.h`（786 行）在 `flash_mla::hytlass_ext`
  命名空间里继承 hute atom 并注册自定义 MMA_Traits——核心是
  `GFX928_16x16x32_F32F16F16F32_NN`/`_BF16BF16F32_NN` 这类 **NN 包装**
  （直接公开继承 NT atom，注释声明 "Preserve FlashMLA's register
  packing without changing HYTLASS's global traits"），即用继承机制把
  上游 kernel 的寄存器打包约定接进 hute 而不动全局 traits。phase1
  kernel 的 TiledMMA、Swizzle、SmemLayout 全部来自 hute。
- **deepgemm-das / lightop / vllm-plugin-das**：无直接依赖（grep 无
  hytlass/hute include）。它们手写 intrinsic（`__builtin_hcu_mmac_*` 等），
  与 hytlass 的 detail 层等价但独立演化。
- 即社区里存在**两条并行路线**：模板库路线（FlashMLA/hute，布局抽象 +
  编译期 tile 组合）与手写路线（aiter/deepgemm/lightop，裸 intrinsic +
  手工流水线）。两路线共享的公共层只有 `__builtin_hcu_*` intrinsic
  本身。

## 对本机优化工作的含义

- hytlass 是 gfx938 HCu intrinsic（mmac/ds_read_matrix/cvt）语义与
  寄存器布局的**权威参考实现**：`mma_traits_gfx928/938.hpp` 里的
  ThrID/ALayout/BLayout/CLayout 特化是 "lane → 矩阵坐标" 映射的
  可执行规格，手写 kernel 对齐布局时可对照（本机已有
  technique-gfx938-instruction-audit 的独立审计，两者可互验）。
- LIT/LTS 位语义（gfx938 保留旧 lane layout 的条件）是跨 928/938 迁移
  kernel 时最容易踩的 ABI 细节，`mmac_gfx9.h:35-36` 注释是社区里对
  该语义最明确的陈述。
- profiler 未默认支持 938 + cluster stub，说明该库对 938 的支持是
  "头文件就绪、工具链半就绪"状态；选用前需自行验证构建。

## 证据限制

- 本页为 source-level 阅读；**没有**本机编译、单测或运行（README 自称
  946 测试通过是上游声明，source-reported）。
- `include/hytlass/` 2.x 层（epilogue/conv/gemm device 层次）与
  `tools/` 未精读；对其描述来自 README/CHANGELOG 与目录结构。
- hute atom 中 `_ALT`/`_FOR_GEMM1` 变体的差异（lane 映射微调）未逐个
  核对；使用时以 traits 特化源码为准。
