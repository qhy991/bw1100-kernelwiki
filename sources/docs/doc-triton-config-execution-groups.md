---
id: doc-triton-config-execution-groups
title: Triton execution-group count must use the target lane width
type: source-doc
architectures: []
tags: [triton, occupancy-tuning, vgpr, lds]
confidence: source-reported
date: '2026-10-07'
url: https://triton-lang.org/main/python-api/generated/triton.Config.html
---

Triton Config将num_warps定义为编译kernel时使用的warp数量；文档以8×32=256线程举例。
示例中的32不是所有target的lane宽度。gfx938实验必须读取实际warp_size及dispatch workgroup，
不能将这个例子照搬成8-wave只有256线程。

改变合作执行的wave数量也会改变每线程工作、数据分配和编译器资源选择；
寄存器更少不自动等于更高整体吞吐。比较时保留tile和语义，检查实际ISA与launch资源，
并在wave总数变化后重新解释按wave归一化的counter；不要只比较归一化数字大小。
本页提供API含义，不给Hygon声明VGPR池、resident wave上限或occupancy常数。
