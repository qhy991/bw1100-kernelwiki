---
id: technique-hcu-layout-contracts
title: HCU 布局契约：跨仓库收拢的 swizzle、重排与 tile 词汇
type: wiki-technique
architectures:
- gfx938
tags:
- lds
- layout-transform
- gemm
- precision
confidence: source-reported
sources:
- doc-hygon-pytorch-int4mm-and-llvm-intrinsics
- doc-hygon-ck-templates
- doc-hygon-moe-aiter-backend
- doc-hygon-tilekernels-tilelang
- doc-hygon-triton-hcu-backend
date: '2026-10-31'
description: 社区实现共享的布局硬约定：LDS major 维 16-Dword wrap、C-tile plain↔lit 重排、SRD swizzle 与 tile 词汇表。
techniques:
- layout-transform
- tiling
hardware_features:
- lds
- mmac
related:
- hw-bw1100-gfx938
---

社区实现共享的布局硬约定：LDS major 维 16-Dword wrap、C-tile plain↔lit 重排、SRD swizzle 与 tile 词汇表。

本页是**源码级事实的横向收拢**（source-reported，无本机实验）：
这些约定散在五个社区仓库的实现里，每条都可在对应 source 页找到
原文定位。指令面能力结论见 [[hw-bw1100-gfx938]]，本页只管布局。

## LDS 矩阵数据布局

- **major 维总是 wrap 在 16 Dwords**：CK traits
  （`hcu_mls_traits_gfx938.hpp`）注释明确 "actual lds layout will
  always be wrapped in 16 Dwords on major dim"。矩阵 tile 进 LDS
  时按这条约束排布，线程级偏移由消费方自行推。
- **TileLang 的 swizzle 标注**：TileKernels 的 split-K GEMM 用
  `tilelang.layout.make_hcu_swizzled_layout(..., major_pack=2)` 声明
  HCU 专属 swizzle（bf16 operand + fp32 统计共享 smem 时分复用）。
  该 swizzle 的完整语义未追进 tilelang 源码，引用时以调用点为准。
- **triton 的 BufferLds**：bank-conflict-aware 共享 encoding，lane
  phase 表（b16: {0,2,8,10,32,34,40,42}，b8: {0,1,8,9,32,33,40,41}）
  来自 gfx936/938 上的 pair-collision 探针实测；b4 的 phase 顺序
  未建模（对应 938 无 b4 路径）。

## C 累加器 plain↔lit 重排

gfx938 上 lit_lts 与非 lit 的 MMAC **数值相同、C-tile lane/分量布局
不同**。pytorch int4mm 在跨 936/938 共存 kernel 的 warp 间归约处给出
直接可引用的换算公式：

```
lit_index = (src_lane & 3) + (((src_lane >> 2) & 3) << 6)
          + ((src_lane >> 4) << 2) + (src_component << 4)
```

迁移任何在 936 非 lit 路径上验证过的 C 消费代码（warp 归约、跨
lane 重组）到 938 lit_lts 路径时，必须先过这层重映射；反向亦然。

## SRD swizzle 与 buffer 描述符

- **tcp_cache_swizzle**（Moe）：构造 buffer 描述符时
  `res[1] += 0x40800000`（SRD bit62 cache swizzle + stride 域）、
  `res[2]=0x80000000`、`res[3]=0x00020000`；stride 有 256/196/128/64
  四档编码表（现版本硬编码 128 档）。与 CK WASP pipeline 的
  CACHE_SWIZZLE（修复 B 流 VL1-Tag stall）同族。
- 手工构造 SRD 时 stride 域与 swizzle 位是**描述符层的独立旋钮**，
  不改变 LDS 矩阵读的 bit31 trans（后者在 soffset 上，见
  [[hw-bw1100-gfx938]] 的指令面事实表）。

## tile 词汇表对照

不同仓库用不同词汇描述同族分块，对照使用时不要混用倍率：

| 层级 | Moe 词汇 | CK atom 层 |
|---|---|---|
| block | BLOCK_MNK 16×128×128 | pipeline 模板参数（WASP 4P+8C 等） |
| warp | WARP_MNK 16×32×64 | warp 级偏移（traits 只给 PackedShape） |
| MMA | 16×16×32 | 12 种 gfx938 MLS atom（32x16_b16 … 128x16_b8） |

pytorch tinygemm 的 ROCm 档则是 m16n16k16（N tile 16、wave64），
与 CUDA 档 m16n8k16 不同——跨框架迁移时 N tile 宽度不默认可移植。

## 使用边界

以上均为源码级约定，本机没有对任何一条做编译/数值/计时验证；
引用某条做设计决策时，回到对应 source 页核对 arch 守卫与调用点，
不要把某仓库某一档的硬编码（如 stride 128 档）当成目标默认值。
