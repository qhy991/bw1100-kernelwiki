---
id: technique-wave-reduction
title: Wave64 上的归约：子组宽度、partials 与实际 shuffle 指令
type: wiki-technique
architectures:
- gfx938
tags:
- reduction
- wave64
- lds
confidence: experimental
sources:
- doc-hip-reduction
- doc-hip-extensions
- doc-amd-wave-builtins
- exp-profiler-skill
- exp-lowlevel-probe-20261006
kernel_types:
- reduction
hardware_features:
- wave64
- lds
---

## 先写清参与集合

固定 256 threads 时，width32 方案产生 8 个 subgroup partials；width64 产生 4 个。
两者都必须在一次 block barrier 后合并全部 partials，不能只读取“每 64 线程的一个值”。
尾部输入用零填充，所有参与 shuffle 的 lanes 执行相同序列。hardware wave_size=64
并不禁止逻辑 width32 分组；也不证明 32 是任意算法的正确分组。

## 成本如何变化

完整 LDS tree 在本次编译中有 9 个静态 s_barrier。两层 shuffle 方案各 1 个 barrier，
却分别有 10/12 条 ds_bpermute_b32（两层各 log2(width) 次）。
这份 DTK emission 的 shuffle 没有变成 DPP；ds_bpermute 仍走数据交换资源。
本次 LDSInsts derived metric 对 width64 反而比 tree 高，不能用“shuffle=没有 LDS 成本”推断。

所有方案的源码都声明 shared float[256]，实际仍分配 1024 bytes。只减少 partials
数量而不缩小声明，不会自动得到更小 allocation。这是下一次有界改写可检验的点。

## 正确性与选择条件

当前仅以有限 dyadic FP32 分布检查行和，不能覆盖任意 cancellation、NaN 或 Inf。
RMSNorm、softmax、Welford 的组合状态也不能直接当一个标量 sum 处理。
同步型 shuffle 在某些 DTK 版本的故障见 exp-profiler-skill；不要为改 mask 而绕过 oracle。

本次 width32/64 均正确，幅度小的计时差异不足以确立通用赢家。
先读每个形状的 A/A 与正反顺序，再决定是否值得改 caller。
新版上游 wave_reduce builtin 和手写 DPP 在 gfx938 的可用性仍未验证。
