---
id: doc-hip-uniform-control-flow
title: Uniform control flow needs a real execution-group condition and includes classification cost
type: source-doc
architectures: []
tags: [correctness, profiling, masking]
confidence: source-reported
date: '2026-10-08'
url: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
---

2026-10-08读取的HIP7.16.0 performance guidelines讨论分支分歧、同步等待、predication与寄存器生命周期。
这些是提出实验的机制线索；分类、数据依赖和控制流本身也属于完整执行成本。
同一wave的lane选择不同路径可能串行执行，编译器也可能采用predicate，必须检查实际产物。

该页以32线程阈值举例，不能原样当作gfx938的wave64一致条件。
`threadIdx.x < 32`在一个64-lane wave内并不一致；这说明条件不同，不单凭源码断言实际branch代价。
本库不继承该页面向AMD的硬件常数、占用率结论或工具可用性。

本机exp-compaction-uniform-20261008由设备数据先算每行Count，再对四行是否含部分命中作program级归约。
对齐后的统一分支仍可能付出跨wave通信与barrier；所有实际资源、计数器和收益由该实验来源拥有。


exp-compaction-granularity-20261008进一步区分一wave内部分类与四wave汇总。
单wave产物消除了LDS/barrier，却同时改变program数量、标量地址形成与部分寄存器分配。
因此即使wave总数接近，也不能只凭无barrier认定完整路径更快；本机四臂配对记录各自分母。
