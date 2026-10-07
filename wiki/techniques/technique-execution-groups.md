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
- doc-triton-config-execution-groups
- exp-execution-groups-20261007
- exp-lowlevel-probe-20261006
- doc-llvm-occupancy-tool
- exp-width-qualification
- exp-fp32-staging
date: '2026-10-07'
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

## 固定MMAC的GEMM补证：位置可以反转选择

exp-execution-groups-20261007保持tile64×64×32与MMAC16，比较2/4/8个wave64。
大形状zero位置2-wave相对4-wave约1.051倍速度，8-wave退化；
guarded位置8-wave相对4-wave约1.233倍速度。资源最小或shape相同都不足以决定组数。
8-wave每线程VGPR较少，但线程数更多、LDS翻倍、global load变窄；没有单一机制归因。

profile必须同时查看原始计数和分母。8-wave的VALUInsts从4-wave的797降至528，
总wave数却翻倍，SQ_INSTS_VALU反而多32.5%；LDS原始计数多69%。
按wave指标下降不表示全kernel工作量下降，也不能仅用VGPR/线程数推算未经校准的驻留。
详见doc-triton-config-execution-groups；其32-lane示例不能替代本机实测wave64。
