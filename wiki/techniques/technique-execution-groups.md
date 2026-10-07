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
- doc-waves-per-eu-hint
- exp-waves-hint-20261007
- exp-gather-mapping-20261007
- doc-triton-thread-layout
- exp-row-mapping-20261007
- exp-route-precision-20261007
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

## 数值相等也有输入范围

exp-route-precision-20261007补测normal、抵消、极小值、跨尺度随机与特殊值，覆盖两个规则shape和一个tail。
本轮g2/g4/g8 MMAC路线在24个输入单元上结果相同；m32 vector-dot在normal/dynamic-range不同，
即便前轮精确dyadic全部通过。不要把“同一数学kernel”或输出FP32当作任意lowering逐bit相等的证明。
本补证没有新增容差或框架接受规则。

## 行归约还要同时看program粒度

exp-row-mapping-20261007固定log-softmax公式，区分每program行数和wave数。
多行并非总分给不同wave：vendor TTGIR在127列用[2,2] wave划分，到129列则用[1,4]。
增加行数可能增加每线程值和跨wave归约数据；source LDS改变也可能仍落在相同512B实际分配粒度。
4097×127多行的完整调用收益与减少program数相容，单wave虽无LDS且VALU更少仍较慢；
4097×1024单wave却优于本轮多行配置。不能独立用wave数量、VALU总数或LDS为零决定调度策略。
原实验保留小行数A/A噪声、数值末位变化和未隔离的访存/驻留原因，不建立默认参数或通用occupancy模型。

exp-gather-mapping-20261007的loss交叉对照中，r1w1相比r4w4在4097×1024获益、4097×127退化。
两者总wave数4100/4097接近，program数却是1025/4097；不能只对齐wave总数就视作等价调度。
单wave去掉workgroup barrier仍可能保留wave内LDS转换；应分别检查LDS allocation、ds指令和s_waitcnt。

## waves_per_eu是另一层编译提示

num_warps决定本例合作线程数；waves_per_eu通过LLVM提示资源优化，不是实测驻留wave数。
exp-waves-hint-20261007的MMAC路线在1/2/4/8下机器指令/描述相同，先过滤重复配置。
高寄存器vector-dot路线的hint8虽把VGPR150降至96，却引入92B private segment和运行时23 spills，
HIP按真实launch LDS的预测仍为4 blocks/CU，完整调用明显变慢。
不要将提示值、资源下降、预测驻留和性能接受合并成一项结论，也不要从该点反推gfx938寄存器池常数。
