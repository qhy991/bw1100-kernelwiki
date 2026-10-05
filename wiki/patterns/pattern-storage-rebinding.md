---
id: pattern-storage-rebinding
title: CUDA graph 缓存：同一 Tensor 换 storage 仍要重算
type: wiki-pattern
architectures:
- gfx938
tags:
- cuda-graph
- correctness
- runtime-guard
- cache-invalidation
confidence: experimental
sources:
- exp-night-exclusions
- exp-width-qualification
date: '2026-10-05'
description: 症状：原numeric rounds通过，但同Tensor.set_(新storage)后输出读旧内容或后一次调用修改旧输出。
symptoms:
- stale-graph-input
- storage-rebinding
- retained-output-mutated
related:
- kernel-bw-rmsnorm
- kernel-bw-linear-attention
- pattern-precision-not-output-only
---

症状：原numeric rounds通过，但同Tensor.set_(新storage)后输出读旧内容或后一次调用修改旧输出。

用原Task/原oracle做fresh call、same-object new storage、retained output和fresh view，先记录最早分歧。
可按current data_ptr与public metadata重capture，或每次copy当前输入到own buffers再replay；这两条都需真实检查。
缺data_ptr字符串不是证据，tensor object id也不充分。持久scratch/graph是运行策略，不能存结果或按输入内容分派。
source-specific排除保留旧160/timing，后继修复用新source重新资格，不能悄悄替换历史report。
当前4×4caller检查仅限预声明smoke，不能声称任意alias/strided/shape ABI。
