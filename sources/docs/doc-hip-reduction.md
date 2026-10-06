---
id: doc-hip-reduction
title: HIP hierarchical reduction
type: source-doc
architectures: []
tags:
- reduction
- wave64
- lds
date: '2026-10-06'
url: https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/reduction.html
confidence: source-reported
---

HIP 7.15.0 reduction tutorial，采集于 2026-10-06。

归约可从每线程累加、block 内 LDS tree，转为 wave shuffle 加少量跨 wave partials。
越界输入用运算单位元，block barrier 必须由相应 block 线程共同到达。
通过改写 thread assignment 减少分歧，仍需检查 LDS 访问；shuffle 需要明确宽度。

本轮只测试整 block 行归约，不采用 last-block/global synchronization 方案。
FP32 加法换序的正确性必须由原任务容差裁决；本轮 dyadic 输入故意隔离通信实现，不能证明任意浮点输入等价。
