---
id: doc-index-constant-lowering
title: Integer index specialization requires a preserved arithmetic domain
type: source-doc
architectures: []
tags: [triton, assembly, tiling, correctness]
confidence: source-reported
date: '2026-10-07'
url: https://triton-lang.org/main/python-api/triton-semantics.html
---

Triton语义文档区分运行时tensor整数除法与纯constexpr计算：混合符号tensor除法按C向零取整，
纯constexpr除法按Python向负无穷取整，取余相应变化。整数除法/取余遵守整数promotion，
不能用一条看似等价的Python表达式或浮点近似无条件替代。

[HIP性能指南](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
讨论运算类型成本及二次幂位操作替代的动机。这提供调查方向，不提供gfx938的周期常数、
完整工作负载收益或许可改变整数语义。

真实已知的shape参数可作为编译期常量，让目标编译器选择常量除法路径；非二次幂也可能有整数乘法/移位改写。
仍要检查当前LLVM/ISA、资源和完整调用。exp-index-specialization-20261007使用正uint32索引，
将符号语义差异排除在比较之外，分别记录127/128/129与实际完整输出证据。
