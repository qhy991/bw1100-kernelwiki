---
id: doc-runtime-division-descriptors
title: Runtime integer division can use precomputed reciprocal descriptors
type: source-doc
architectures: []
tags: [assembly, triton, correctness, tiling]
confidence: source-reported
date: '2026-10-07'
url: https://libdivide.com/
---

libdivide为运行时才知道、但会重复使用的整数除数预计算乘数与移位描述，随后用乘法/移位计算商。
其性能说明针对相应CPU/SIMD实现，不提供Hygon性能保证；本轮不引入库依赖，也不声称编写了libdivide。

[读取的上游头文件](https://raw.githubusercontent.com/ridiculousfish/libdivide/master/libdivide.h)
标为5.4.0。unsigned32 branchfree路径使用乘积高位、避免溢出的加法修正及移位。
该生成路径拒绝除数1，通用生成拒绝0；二次幂描述的shift需要补偿执行公式内已有的一次右移。
带分支与branchfree描述格式/规则不同，不能混用其字段或把所有除数都套成一个未验证公式。

[Triton umulhi](https://triton-lang.org/main/python-api/generated/triton.language.umulhi.html)
提供对应宽度乘积的高半部分。本机exp-runtime-divider-20261007用独立Python整数生成与Triton算术实现
一个受限unsigned32形式，先检查CPU描述和设备商余数，再检查共享kernel的完整转置。
预计算、参数传递、代码复用和完整调用是不同成本边界，去掉设备除法不自动带来净收益。
