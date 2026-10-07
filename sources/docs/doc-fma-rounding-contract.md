---
id: doc-fma-rounding-contract
title: FMA and separate multiply-add have different rounding contracts
type: source-doc
architectures: []
tags: [precision, fp32, triton, correctness]
confidence: source-reported
date: '2026-10-07'
url: https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html
---

CUDA浮点说明区分一次舍入的FMA与先乘、再加的两次舍入，解释了抵消时的不同结果。
这里只采用舍入数学解释，不将CUDA编译开关、吞吐或特殊值行为作为gfx938事实。

[Triton fma API](https://triton-lang.org/main/python-api/generated/triton.language.fma.html)
提供显式融合乘加；隐式表达式收缩开关与显式操作不是同一个合同层。
exp-fp-contraction-20261007用本机ISA与独立整数舍入参考验证它们，配置值均显式指定，不继承其他版本默认值。

与实数表达式更接近不自动满足一个要求分步FP32舍入的reference。
允许收缩、要求分步舍入、非有限分类和NaN payload应由Task说明；本机有限输入观察不授予一般特殊值资格。
