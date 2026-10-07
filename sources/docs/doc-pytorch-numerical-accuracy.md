---
id: doc-pytorch-numerical-accuracy
title: Floating point ordering, reference precision and target-specific denorms
type: source-doc
architectures: []
tags: [precision, fp32, fp16, correctness]
date: '2026-10-06'
url: https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html
confidence: source-reported
---

PyTorch2.14数值说明，2026-10-06读取。浮点运算顺序会影响结果，数学等价不保证CPU/GPU
或不同实现的bitwise一致。高精度reference的舍入结果和一次FP32逐步累加是不同问题。

文档另有MI200特定FP16/BF16指令denormal行为及相关库路径说明。该结论有明确设备、
指令和库范围，不能因gfx名字相近而套给Hygon MMAC。本机subnormal探针应单独留证据。
原Task仍拥有dtype、中间舍入、容差与特殊值规则；此页不授予修改这些规则的权限。

2026-10-07复查该说明与[rocBLAS6.2的MI200范围说明](https://rocm.docs.amd.com/projects/rocBLAS/en/docs-6.2.0/how-to/what-is-rocblas.html)。
本机后继exp-bf16-numerical-20261007验证两个gfx938 BF16路径的指定subnormal输入与输出保留，
仍将上游设备特定声明与本机有限观察分开；没有改写成全面FTZ或cast舍入保证。

2026-10-07另读取[与本机版本对应的PyTorch2.11数值说明](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)。
其非结合性与跨实现不保证逐bit相等的说明由本机exp-atomic-numerical-20261007作有界补证；
该说明不能用来忽略具体Task的误差接受条件。
