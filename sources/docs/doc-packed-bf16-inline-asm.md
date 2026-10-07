---
id: doc-packed-bf16-inline-asm
title: Packed BF16 conversion needs target admission and real per-thread operands
type: source-doc
architectures: []
tags: [assembly, bf16, precision, triton, correctness]
confidence: source-reported
date: '2026-10-07'
url: https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html
---

Triton inline_asm_elementwise API说明pack是每次asm处理的元素数，具体输入集合的映射未指定；
不足4-byte的输入会打包到32-bit寄存器。约束必须匹配实际target汇编，is_pure声明无副作用。
这不承诺自动跨lane收集数据，也不授予目标对某条指令的支持。

[LLVM PR116678](https://github.com/llvm/llvm-project/pull/116678)在2024年加入gfx950的
V_CVT_PK_BF16_F32支持，并包含相关编码与转换测试。
[AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
也列出该指令；这两份上游AMD资料不直接描述Hygon gfx938。

本机exp-packed-bf16-20261007先在gfx938编译器准入，再检查有效操作数、有限舍入边界、特殊值与尾部。
出现packed opcode不表示两个输入都有效；编译通过也不表示语义或性能已接受。
必须保留具体布局、寄存器打包与输出位模式证据，不能仅凭名字或别的target表替代验证。
