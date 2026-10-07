---
id: doc-llvm-denormal-modes
title: Denormal input and output modes are separate compiler assumptions
type: source-doc
architectures: []
tags: [precision, fp32, triton, correctness]
confidence: source-reported
date: '2026-10-07'
url: https://releases.llvm.org/17.0.1/docs/LangRef.html#denormal-fp-math
---

LLVM17归档LangRef说明旧式denormal-fp-math属性：第一项为输出模式，第二项为输入模式，
省略第二项时两者采用同值；f32专用属性可覆盖32位浮点行为。
输入preserve-sign/positive-zero要求算术把denormal当零；输出模式允许清零，但不要求每个结果都清零。
属性本身也不保证运行环境已设置成对应模式。

本页用归档文档解释本机实际出现的字符串属性，不把外部clang版本等同于Triton内嵌LLVM版本。
上游当前主线的属性表示已演进，不能只在最新文档中搜不到旧拼写就判断本机未使用它。

exp-denorm-policy-20261007分别检查编译选项、LLVM属性、HSA字段和设备模型匹配。
确定性输入/输出模型用于区分行为，不是LLVM允许行为的穷举，更不是所有gfx938 kernel的默认保证。
