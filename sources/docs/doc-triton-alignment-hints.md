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
