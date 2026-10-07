---
id: doc-triton-tensor-gather
title: Tensor gather specifies selected values, not a free register lookup
type: source-doc
architectures: []
tags: [triton, reduction, lds, vgpr, correctness]
confidence: source-reported
date: '2026-10-07'
url: https://triton-lang.org/main/python-api/generated/triton.language.gather.html
---

Triton main的gather API按src、index、axis定义从tensor选取值。
该语义不保证只访问某个线程的本地register，也没有免除分布式tensor的跨lane/wave通信。
具体实现取决于当前编译器、tensor布局、轴与索引形状；需结合doc-triton-thread-layout读取真实lowering。

exp-target-selection-20261007比较global重读、掩码选择归约和tl.gather，记录了本机完整tensor布局转换及LDS代价。
本机vendor Triton3.6.0的结果不能由上游main API文字直接推得，也不推广为gather总比reload差。
