---
id: doc-triton-atomic-reduction
title: Atomic reduction changes require order, scope and returned-value contracts
type: source-doc
architectures: []
tags: [precision, correctness, triton, host-overhead]
confidence: source-reported
date: '2026-10-07'
url: https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html
---

Triton atomic_add返回更新前的值；默认sem为acq_rel，默认scope为gpu，接口允许显式指定relaxed等语义。
把多个atomic先求和再更新，只有在返回值不被消费且任务允许改变归约顺序等条件下才可能保留合同。
不能将仅最终sum输出的reduction结论转给需要每线程旧值的fetch-add。

[tl.sum说明](https://triton-lang.org/main/python-api/generated/triton.language.sum.html)要求归约操作可结合、可交换。
浮点加法对任意输入不满足实数式结合性；本机策略探针使用可精确表示的有界dyadic数据，
借此隔离冲突与提交成本，不以该输入证明通用FP32精度等价。
实际atomic可能是CAS循环或带地址聚合的路线，必须检查emission，不能仅凭API名估算硬件atomic数量。
