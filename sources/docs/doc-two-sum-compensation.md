---
id: doc-two-sum-compensation
title: TwoSum error terms do not automatically qualify an arbitrary reduction tree
type: source-doc
architectures: []
tags: [precision, correctness, fp32]
confidence: source-reported
date: '2026-10-07'
url: https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf
---

Ogita、Rump、Oishi，Accurate Sum and Dot Product，SIAM J. Sci. Comput.26(6)，2005。
论文的TwoSum将浮点和及其误差项分开，并在明确假设下分析级联求和。
相关定理并不自动覆盖任意GPU并行树、编译器重结合、溢出或非有限输入。
论文也给出补偿不必对每个输入单调改善误差的讨论。

本机探针采用TwoSum派生的hi/lo pair合并，并用Triton tuple reduce构成树，
不是逐行复现论文Sum2/SumK，也不继承那些算法的完整误差界。
先检查实际加减指令与优化标志，再验证有限分布、特殊值和成本；不能用论文标题替代本机资格。
