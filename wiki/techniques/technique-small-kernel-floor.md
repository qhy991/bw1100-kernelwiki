---
id: technique-small-kernel-floor
title: 短 kernel 先量时延下限，再决定改访存宽度还是并行度
type: wiki-technique
architectures: [gfx938]
tags: [launch-configuration, occupancy-tuning, copy, profiling, rocprof, hip, negative-result]
confidence: experimental
sources:
- exp-training-native-probes-20261028
- exp-tail-vectorization-20261007
related:
- technique-bulk-tail-vectorization
- technique-host-entry
- technique-profile-gfx938
techniques: [launch-configuration, occupancy-tuning]
hardware_features: [wave64]
kernel_types: [activation, reduction]
---

## 主要代价

数据只有几十 KB 到一两 MB 时，一次 dispatch 的设备时长主要由固定部分决定：启动 wave、
首批访存的往返延迟和收尾，而不是带宽或算术吞吐。exp-training-native-probes-20261028 在同一
形状上测到空 kernel 2.36 μs、只读 2.94 μs、只写 3.11 μs、标量拷贝 3.18 μs、vec4 拷贝 3.23 μs；
多个 elementwise/拷贝 task 的最佳实现落在 3.4–4.5 μs，而换访存宽度、加 nontemporal store、
改 launch bounds 后的手写版本都落在同一带内或更慢。

## 改写顺序

1. 先写一个与目标 kernel 同 grid、同字节数的空 kernel、只读 kernel 和只写 kernel，用 rocprof
   时间戳测中位数。这给出"不可能更低"的下限。
2. 目标 kernel 离下限不到约 1 μs 时，停止在宽访存、nontemporal、除法改乘法、查表等单指令级改写上
   投入；这些改写在该区间内多次测得不变或变慢。
3. 离下限较远时，先看并行度：本批 adamw(E 为数万)用 1024 个 CTA 的标量写法优于 128/256 个 CTA
   的 float4 写法，momentum_sgd 的更大 grid 反而更慢。合适的 CTA 数比每线程访存宽度更早决定结果。
4. 再看访存是否真的是宽指令。exp-tail-vectorization-20261007 表明大数组上恢复 dwordx4 能带来约
   1.6 倍；在本页的小数组上同样的改写没有稳定收益。

## 条件与可能退化

- 下限随 grid、wave 数和 host 提交方式变化；每个形状都要单独测，不能沿用别的 task 的数字。
- 减少 CTA 数(更宽 tile、每线程更多元素)会降低可并行隐藏的延迟；在下限区间里常见的是变慢。
- 单指令计数下降(例如 rcp 替代除法使 v_div_* 从 20 降到 0)不等于时长下降，要用时间戳确认。
- 设备时长接近下限时，callable 总时间通常由 host 包装决定；那时优化应转向 host 入口
  (technique-host-entry)，而不是 kernel 体。

## 本机范围

证据是 gfx938、DTK hipcc -O3 与 vendor Triton 3.6.0 下 20 余个 task 的单形状测量，手写与生成 kernel
分别在不同 admission 中计时，没有 ABA/BAB 配对。小于约 0.3 μs 的差距不作排序。未测量大数组上的
同一组改写，也未建立以字节数划分区间的阈值。
