---
id: doc-llvm-pointer-alignment
title: Pointer alignment attributes are caller facts, not memory repair
type: source-doc
architectures: []
tags: [correctness, runtime-guard, triton]
date: '2026-10-06'
url: https://llvm.org/docs/LangRef.html#parameter-attributes
confidence: source-reported
---

LLVM在线LangRef，2026-10-06读取；不是本机vendor compiler的版本锁定实现说明。
align可以逐pointer参数表达保证，值必须是合法的power of two。违反该属性会破坏IR合同，
并不要求compiler帮caller重新分配或修正地址。

因此A、B、C的事实应独立保留。只知道某一个pointer偏移，不构成抹去其他pointer事实的理由；
也不能将更强属性填入不满足的pointer。具体Triton版本的attribute编码仍由实际工具链检查。
现有Cake pointer_alignment_attributes已经是其合同owner；本页不建立第二个声明表。
本来源没有证明任何gfx938速度收益。本机逐operand的实际验证另见exp-gemm-operand-alignment-20261006。
