---
id: doc-triton-dot-precision
title: Triton dot input types and precision controls
type: source-doc
architectures: []
tags: [precision, triton, gemm, mmac]
date: '2026-10-06'
url: https://triton-lang.org/main/python-api/generated/triton.language.dot.html
confidence: source-reported
---

Triton main dot API，2026-10-06读取。input_precision描述F32×F32的计算选项；
文档指出非F32输入不受该选项控制。输入、accumulator和output dtype仍需分别检查。

不要看到IR里一个tf32文字标签就断定F16输入已经走TF32；本机需要结合typed operands、
有效metadata、实际MMAC opcode与数值实验解释。本轮使用F16 operand和F32 accumulator/output。
FP32 accumulation也不等于正确舍入的实数矩阵乘法，尤其在大数抵消时。
