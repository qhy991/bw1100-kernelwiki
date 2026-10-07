---
id: doc-stall-counter-domains
title: Stall counters have interface, aggregation and normalization domains
type: source-doc
architectures: []
tags: [profiling, rocprof, correctness]
confidence: source-reported
date: '2026-10-07'
url: https://rocm.docs.amd.com/en/docs-6.0.0/conceptual/gpu-arch/mi200-performance-counters.html
---

ROCm MI200 counter表分别列出TCP数据接口、读/写tagram冲突，以及TCC到EA写请求stall。
这些事件位于不同接口，cycle计数也可能按多个实例汇总，不是可直接相加的调用损失时间。
这是上游AMD定义，不授予gfx938相同窗口、实例数量或时钟域事实。

[ROCm profiling说明](https://rocm.blogs.amd.com/software-tools-optimization/roc-profiling/README.html)
给出MemUnitStalled等派生公式，包含原始最大值、GRBM active与SE_NUM。
按名称理解“内存单元等待百分比”不够，必须核对本机真实表达式及分子、分母和窗口。

exp-transpose-stalls-20261007发现本机TCP数据接口事件标为Not Windowed，保留该范围限制；
对WriteUnitStalled同pass核对原始最大值与active分母，另列第二写接口，不猜SE_NUM或把零值当作无等待证明。
百分比、最大值和实例sum各有自己的统计域，不能混成roofline或直接解释wall差异。
