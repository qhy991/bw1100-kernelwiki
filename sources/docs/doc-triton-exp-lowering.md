---
id: doc-triton-exp-lowering
title: Triton exponential APIs and approximate softmax ingredients
type: source-doc
architectures: []
tags: [triton, precision, profiling]
confidence: source-reported
date: '2026-10-07'
url: https://triton-lang.org/main/python-api/generated/triton.language.exp.html
---

Triton exp计算自然指数，[exp2](https://triton-lang.org/main/python-api/generated/triton.language.exp2.html)计算以2为底的指数。
[fused softmax教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)明确把Triton指数描述为快速近似运算。
数学恒等式exp(x)=exp2(x·log2(e))不保证两个不同实现逐位相同，也不保证手写后更快。

本机exp-exp2关系、设备数学库symbol、边界行为和指令成本由exp-exp-route-20261007的
实际vendor编译/设备观察决定。上游CUDA类比不直接证明gfx938 SFU精度或denormal政策。
不要把库函数名字当作正确舍入承诺，也不要把单独exp吞吐当作softmax或GELU端到端收益。
