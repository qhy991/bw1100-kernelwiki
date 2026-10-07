---
id: doc-argmax-tie-contract
title: Argmax owns both value ordering and the tie-breaking index contract
type: source-doc
architectures: []
tags: [triton, reduction, int32, correctness]
confidence: source-reported
date: '2026-10-08'
url: https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/python/triton/language/standard.py
---

Triton v3.6.0的_argmax_combine在tie_break_left开启时，同时比较value与index：较大value胜出；value相同时较小index胜出。
随后用同一个选择条件返回value与index。argmax只返回index，但max(return_indices=True)也保留值。
[当前API说明](https://triton-lang.org/main/python-api/generated/triton.language.argmax.html)将左侧tie规则限定到非NaN值；
本库的INT32实验没有浮点NaN或signed-zero问题，不因此推断浮点合同。

[NumPy argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)说明并列最大值返回首次出现位置，
用于本机独立oracle；device结果还必须验证最大值与该位置匹配。输出一个合法最大值不足以验证索引。

exp-argmax-key-20261008另行构造有符号值与反向索引的uint64顺序键，并与两字段归约对照。
该编码是本地候选，不是上述上游实现或官方性能保证；符号位变换、索引方向、padding中性元及整数宽度都属于需要证明的条件。
