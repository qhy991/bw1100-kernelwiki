---
id: kernel-bw-convnext-grn
title: ConvNeXtV2 / GRN：绑定权重与 read-only image cache
type: wiki-kernel
architectures:
- gfx938
tags:
- conv
- normalization
- fp32
- runtime-guard
confidence: experimental
sources:
- exp-community-baselines
- exp-admission
- doc-hygon-pytorch-int4mm-and-llvm-intrinsics
date: '2026-10-31'
description: ConvNeXtV2/GRN的优化空间来自卷积、归约与缩放之间的中间数据和调用边界。
kernel_types:
- conv
- normalization
languages:
- python
- triton-rocm
related:
- pattern-jit-cache
- pattern-hcu-release
---

ConvNeXtV2/GRN的优化空间来自卷积、归约与缩放之间的中间数据和调用边界。

## 优化思想
先让社区block使用提供的权重。GRN等归约/elementwise若能复用卷积输出tile或相邻buffer，可减少临时materialization；布局合适时也能降低后续访存成本。

## 条件与代价
卷积和全域归约的tile域未必对应，跨tile统计量可能需要额外阶段。融合带来的resource footprint应与减少的内存往返一起比较，并保留FP32卷积/GRN语义。

## 本机范围
已有timm block资格，cache路径故障属于环境问题。这里不填写尚未确认的融合收益；后续应按完整block计时。

## 社区对照（源码，非本机实测）
52 个 HYGON-AI 社区仓库扫描里**没有**视觉域（ConvNeXt/GRN 类）的
专门实现（收尾扫描见
doc-hygon-pytorch-int4mm-and-llvm-intrinsics：TurboPhysAI 仅一个
upsample kernel，spconv-das 只有 builtin 别名 shim）。本题在社区
基线里没有直接对标物——GRN 归约、卷积 tile 化都要按本页机制自行
设计，只能借用通用机制层（MMAC/LDS 布局契约）。
