---
id: doc-triton-alignment-hints
title: Triton alignment and contiguity are compiler promises
type: source-doc
architectures: []
tags: [triton, tiling, correctness]
date: '2026-10-06'
url: https://triton-lang.org/main/python-api/generated/triton.language.multiple_of.html
confidence: source-reported
---

Triton main API文档，2026-10-06读取。multiple_of表达每个contiguous group起点的可整除信息；
[max_contiguous](https://triton-lang.org/main/python-api/generated/triton.language.max_contiguous.html)
表达沿维度的连续长度。这些信息可使compiler采用依赖对齐的向量化。

这类hint是作者对输入的承诺，不是运行时修复地址。不能把torch base pointer对齐直接推广为
每一行或任意view都对齐。AOT参数attribute需要当前版本的实际编码和caller检查。
本机实验只对A/B/C基址声明16-byte divisibility，并检查data_ptr()%16，保留odd-stride反例。

2026-10-08重读[Gluon multiple_of](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.multiple_of.html)：
它描述contiguous group起点的整除信息，contiguity为1时可按每个元素的倍数事实理解。
exp-argmax-peel-20261008只对由整数公式保证为4倍数的bulk起点和长度作承诺，并把padding行也纳入CPU证明。
源码有hint还不够：本机一个减零表达式折叠后，最终IR中起点标注未保留；避开身份运算再标注才实现两种offset的向量load。
这保留为当前vendor编译链观察，不声称上游版本都存在同一问题，也不授权对未对齐指针直接声明16B。
