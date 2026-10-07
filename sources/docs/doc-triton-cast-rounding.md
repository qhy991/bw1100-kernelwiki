---
id: doc-triton-cast-rounding
title: Triton downcast rounding modes and special-value boundaries
type: source-doc
architectures: []
tags: [precision, bf16, fp32, correctness, triton]
confidence: source-reported
date: '2026-10-07'
url: https://triton-lang.org/main/python-api/generated/triton.language.cast.html
---

Triton cast API的fp_downcast_rounding参数用于浮点向较窄类型转换，支持rtne与rtz。
[上游v3.6.0语义实现](https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/python/triton/language/semantic.py)
把未指定的浮点downcast默认设为RTNE，并让自定义非RTNE走独立fp-to-fp lowering。
这说明不同模式可能由不同后端路径实现，不保证本机产生相同指令成本或相同特殊值处理。

RTNE的有限数ties-to-even与RTZ的截断是不同数值合同，不能因某种指令更简单就替换。
NaN、Inf、signed zero、溢出与subnormal需要分开检查，尤其基于bit截断的实现。
exp-bf16-cast-20261007记录vendor Triton3.6.0/gfx938实际转换及RTZ低payload NaN分类变化。
该实验不替上游API定义未声明的NaN payload或异常标志政策。


exp-rounded-consumer-20261007把显式RTNE用于可见BF16复制，验证正确内部转发需先窄化再加宽。
cast API表达舍入操作，不授权消费者绕过它；仅输出BF16位模式正确不足以接受整个融合图。
