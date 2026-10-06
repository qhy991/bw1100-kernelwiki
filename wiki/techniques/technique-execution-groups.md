---
id: technique-execution-groups
title: 执行组选择：工具可复用，参数需要测量
type: wiki-technique
architectures:
- gfx938
tags:
- execution-groups
- vgpr
- scratch
- negative-result
confidence: experimental
sources:
- exp-lowlevel-probe-20261006
- doc-llvm-occupancy-tool
- exp-width-qualification
- exp-fp32-staging
date: '2026-10-05'
description: 执行组数量是在同一逻辑计算与不同物理线程分摊之间作选择。
techniques:
- launch-configuration
- occupancy-tuning
hardware_features:
- vgpr
- scratch-memory
- lds
related:
- kernel-bw-gateup
- kernel-bw-moe-fp32
- pattern-version-comparison
---

执行组数量是在同一逻辑计算与不同物理线程分摊之间作选择。

## 优化思想
把一块逻辑tensor分给更多执行组，可减少每线程持有的元素和accumulator；它也可能增加跨wave通信、同步及每CTA线程占用。更少的组则可能增加register或scratch需求。

## Compiler改写
显式width pass保留数学body、tile、K循环、precision、access maps与runtime有效域，只改execution_groups及结果身份。Target决定lane width和上限，候选重新assess。固定load/widen/MMA域不扩大为任意state/sync/persistent程序。

## 选择方法
先比较实际编译/dispatch的VGPR、LDS和scratch，再用同Workload的完整调用配对判断。资源较少并不单独证明延迟较小，静态pressure也不等于physical allocation。

## 本机观察
width2在一个GateUp域中数值正确但scratch较大、完整调用较慢；width8的scratch为0。这支持把width作为显式搜索参数，并保留负结果，不支持默认少线程或固定8组。原source保存条件和原始资源。

底层检查见 technique-gfx938-instruction-audit：本轮 source metadata 的 VGPR/LDS 与
profiler allocation 存在粒度差异；只改线程分组或 partial 数量不能保证 allocation 下降。
