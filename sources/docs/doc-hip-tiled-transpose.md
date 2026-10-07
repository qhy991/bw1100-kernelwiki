---
id: doc-hip-tiled-transpose
title: HIP tiled transpose and coalescing
type: source-doc
architectures: []
tags: [tiling, layout-transform, lds]
date: '2026-10-06'
url: https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/docs-7.2.1/tutorial/hip-performance-optimization/tiling-matrix-transpose.html
confidence: source-reported
---

AMD ROCm Programming Guide 7.2.1，采集于 2026-10-06。
教程以 transpose 说明输入连续、输出跨行时的事务成本，以及通过 shared tile 将两侧访存分开的办法。
这是机制来源，不是 gfx938 bank、向量宽度或最佳 tile 的硬件规格。

本轮后继把现有原生探针推广到长宽互换与非整 tile 边界，显式使用输入 cols 与输出 rows
作为各自 leading dimension。相同元素数并不代表相同访问 stride 或相同速度。
只验证 contiguous/out-of-place FP32，不包括 in-place 或 arbitrary-strided tensor。


exp-transpose-access-20261007另用冻结int32位模式比较gather/scatter与Triton二维分块，
显式覆盖编译器生成的布局转换。该后继不继承早期FP32方阵的最优选择，且不因LDS出现就认定全局访问已合并。
