---
id: technique-existing-layout-maps
title: 先用现有 access maps 表达有界 memory permutation
type: wiki-technique
architectures:
- gfx938
tags:
- layout-transform
- copy
- correctness
confidence: experimental
sources:
- exp-register-values
date: '2026-10-05'
description: 布局优化先问能否通过已有读写索引改变访问，而不是先增加一种layout抽象。
techniques:
- layout-transform
related:
- kernel-bw-rope
- technique-rounded-tiled-fusion
---

布局优化先问能否通过已有读写索引改变访问，而不是先增加一种layout抽象。

## 优化思想
把输入元素到输出元素的映射写进load/store/access_maps，可以把copy或permutation与邻近计算结合。选择coalesced方向时，还要考虑后续consumer需要的布局。

## 条件与代价
数学元素对应、mask、alias与全域coverage必须保留。消除一次transpose可能让另一阶段访问更差；register permutation与global memory permutation不是同一机制。

## 本机范围
一个BF16memory copy有界bitwise通过，说明该映射可由现有primitive表达。完整RoPE、任意permutation和性能效果仍是独立问题。
