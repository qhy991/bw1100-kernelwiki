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
