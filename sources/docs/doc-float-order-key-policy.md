---
id: doc-float-order-key-policy
title: Floating-order transforms do not choose NaN policy or preserve a canonicalized payload
type: source-doc
architectures: []
tags: [fp32, correctness, reduction]
confidence: source-reported
date: '2026-10-08'
url: https://raw.githubusercontent.com/NVIDIA/cccl/main/cub/cub/device/device_radix_sort.cuh
---

2026-10-08读取的CCCL主线DeviceRadixSort注释解释了浮点排序变换：正数翻转符号位，负数翻转全部位；正负零作为相等处理。
它明确NaN不作特别处理，而按变换后的位表示排序。主线是可变来源；这些说明不构成Hygon支持、性能或argmax首次NaN合同。

exp-argmax-fp-key-20261008另行声明“有NaN选首次NaN，否则选首次数值最大值，保留所选原始位模式”。
因此候选对NaN和正负零的排序键做显式归一，但输出从获胜索引回读原始输入；排序键的等价类不等于可丢弃的输出信息。
其padding、subnormal、NaN sign/payload和返回位模式资格由本机实验拥有，不把sort语义直接套到argmax。
