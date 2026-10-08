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
- exp-argmax-compact-binding-20261008
- exp-argmax-rebind-20261008
- exp-argmax-anchor-cpu-20261008
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


## 数据 anchor 的 CPU 重绑定边界

exp-argmax-anchor-cpu-20261008在目标机现有Torch的CPU路径上验证：同一Tensor换storage后，旧anchor仍持有旧storage。
每次从当前view重新构造anchor可读回当前全部bits；显式as_strided偏移必须相对storage计算。
32组输入和两类错误控制保留，未运行GPU或测量调用成本；不把此检查当作第七十轮固定绑定收益的新资格。


## 重绑定成本必须进入完整调用

exp-argmax-rebind-20261008复用第七十轮kernel，验证同一Tensor换storage后的全部输出和旧结果。
三条直接配对分别比较Torch、计时外绑定及每call绑定；32组native已测指令相同，标记区间没有额外copy。
每call验证和view重建增加约13μs提交成本，大N1024偏移场景对Torch由固定绑定约1.50倍缩至约1.16倍。
小batch与N129仍落后；保留A/A波动，不把CPU元数据正确性或固定绑定收益当作通用adapter资格。


## 先减少当前调用的metadata工作

exp-argmax-compact-binding-20261008把view加slice合为一次合法as_strided，并复用本次读取的pointer值；仍检查dtype、shape、连续性及anchor地址关系。
每次从当前storage重建，无跨调用数据缓存；CPU等价/拒绝、实际GPU输出与旧结果检查保持。
组合改写减少约4.7–5.6μs提交成本，原生完整调用约1.14–1.18倍改善；大N1024偏移域对Torch确认约1.32–1.34倍，小shape仍慢。
两项host改动的独立贡献未拆分；计时与A/A、错误比较臂前驱均保留，不把有效trace或输出正确当作预期候选已运行的充分证据。
